"""
cmr_reasoner.py - Port cua CMR (Concept-based Memory Reasoner, Debot et al.,
NeurIPS 2024, "Interpretable Concept-Based Memory Reasoning"). Nguon:
github.com/daviddebot/CMR, file experiments/mnist/models_copy.py (class
MNISTModel + CMR + RuleModule + ProbRDCat + InputTypes), da xac nhan qua
doc code goc la LOP TONG QUAT THEO DATASET (ho dung y het class nay cho ca
MNIST/CUB/CelebA, chi doi encoder/hyperparam), khong phai thu hard-code
rieng cho MNIST du nam trong thu muc do.

Muc tieu: giu NGUYEN VEN thuat toan cot loi (forward, training_step,
validation_step, aggregate_rules, predict, predict_concepts) tu ban goc --
khong sua logic, chi doi ten file/vi tri de tich hop vao repo nay. Cac
phan CO TRONG ban goc nhung KHONG duoc port vi da xac nhan la dead code /
tinh nang khong dung toi trong recipe huan luyen chuan (kiem tra ky truoc
khi bo, khong doan):
  - `ConceptEmbedding`/`c_emb_combiner` trong `MNISTModel.__init__`: duoc
    khoi tao nhung KHONG BAO GIO duoc goi trong forward() -- da doc toan
    bo 768 dong file goc de xac nhan, khong chi mot phan.
  - `add_rules`/`add_rules_irr`/`mask_rule`/`get_added_rule_probs`/
    `combine_added_and_learned`/`check_polarity_crispness`: tinh nang
    "RuleAdd Intervention" thu cong, khong duoc goi trong recipe train/eval
    chuan (vd script CUB cua chinh ho). `get_all_rule_vars()` van hoat dong
    dung khi khong port cac ham nay, vi no chi kiem tra `self.added_rules`
    (luon la [] neu khong goi add_rules).
  - `calc_avg_p_c_rec`: khong duoc goi trong training_step/validation_step/
    aggregate_rules cua recipe chuan.
  - `SaveBestModelCallback`/`SaveBestModelCallbackVal2`: 2 bien the
    callback khac cua ho, script CUB chi dung `SaveBestModelCallbackVal`
    (theo dong val_loss) nen chi port dung ban nay.

Neu can doi chieu lai voi ban goc (vd nghi ngo sai lech), xem dung file
experiments/mnist/models_copy.py tren GitHub cua ho, khong phai file nao
khac trong repo (celeba/cub/... deu import lai chinh class nay).

Khac biet so voi ban goc (chi 2 diem, ca hai deu khong dung cham thuat
toan): (1) khong con phu thuoc `utils.logic.ConceptEmbedding` (da xac nhan
dead code o tren) nen bo import do; (2) `save_hyperparameters()` doi thanh
`save_hyperparameters(ignore=["encoder", "rule_module"])` -- ban goc luu
ca object encoder (nn.Module) va class rule_module vao hparams, gay canh
bao/loi khi Lightning co gang serialize checkpoint; bo qua 2 truong nay
KHONG anh huong training/inference, chi anh huong metadata luu trong
checkpoint.
"""
from __future__ import annotations

import copy
from collections import defaultdict

import torch
import lightning.pytorch as pl
from sklearn.metrics import accuracy_score

from torch.nn.functional import binary_cross_entropy

EPS = 1e-18
CONCEPT_EMB_SIZE = 16


def reasoning(logic, concepts, polarity, relevance):
    pospolarity = polarity  # batch, task, rule, concept
    irrelevance = 1 - relevance
    negpolarity = 1 - pospolarity - irrelevance

    # avoid floating point errors resulting in > 1 probabilities
    pospolarity = 0.999 * pospolarity
    negpolarity = 0.999 * negpolarity
    irrelevance = 0.999 * irrelevance

    preds = irrelevance + (1 - concepts) * negpolarity + concepts * pospolarity
    return torch.prod(preds, dim=-1)


class InputTypes:
    """Possible inputs for the rule selector."""
    concepts = 0
    embedding = 1
    concepts_ground_truth = 2


class SaveBestModelCallbackVal(pl.Callback):
    """Callback theo val_loss -- dung khi train (checkpoint theo epoch tot nhat)."""

    def __init__(self):
        super().__init__()
        self.best_loss = float("inf")
        self.best_state_dict = None
        self.best_epoch = 0

    def on_validation_end(self, trainer, pl_module):
        if "val_loss" in trainer.callback_metrics:
            val_loss = trainer.callback_metrics["val_loss"]
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_state_dict = copy.deepcopy(trainer.model.state_dict())
                self.best_epoch = trainer.current_epoch


class RuleModule(torch.nn.Module):
    def __init__(self, rule_emb_size, n_tasks, n_rules):
        """
        Abstract class for rule modules. A rule module stores rule embeddings, and
        provides (1) a way to decode them into polarities and relevances, (2) a
        way to predict the task given the symbolic rules, and (3) a way to
        compute the 'concept reconstruction'.
        """
        super().__init__()
        self.rules = torch.nn.Embedding(n_tasks * n_rules, rule_emb_size)
        self.n_rules = n_rules
        self.rule_emb_size = rule_emb_size

    def decode_rules(self, rule_embs):
        raise NotImplementedError

    def calc_y(self, c, pospolarity, relevance):
        return reasoning(self.logic, c, pospolarity, relevance)

    def calc_c_rec(self, pospolarity, relevance):
        raise NotImplementedError


class ProbRDCat(RuleModule):
    def __init__(self, rule_emb_size, n_concepts, n_tasks, n_rules):
        """
        A rule module where rules are decoded into a categorical variable defining
        positive polarity, negative polarity, and irrelevance. Therefore, they
        are mutually exclusive.
        """
        super().__init__(rule_emb_size, n_tasks, n_rules)
        self.logic = None
        self.n_concepts = n_concepts
        self.rule_emb_size = rule_emb_size
        self.rule_decoder = torch.nn.Sequential(
            torch.nn.Linear(self.rule_emb_size, self.rule_emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.rule_emb_size, 3 * self.n_concepts),
        )

    def decode_rules(self, rule_embs):
        shape = rule_embs.shape[:-1]
        shape += (self.n_concepts, 3)
        logits = self.rule_decoder(rule_embs).view(shape)
        return torch.softmax(logits, dim=-1)

    def calc_c_rec(self, pospolarity, relevance):
        return 0.5 * (1 - relevance) + pospolarity


class MNISTModel(pl.LightningModule):
    """Ten giu nguyen tu ban goc (ho dung chung class nay cho moi dataset,
    khong rieng MNIST) de de doi chieu neu can kiem tra lai voi source."""

    def __init__(self, encoder, emb_size, rule_emb_size, n_tasks, n_rules, n_concepts, concept_names, rule_module,
                 lr=0.001,
                 selector_input=InputTypes.concepts,
                 rel_concept_counts=None, weight_concepts=False,
                 w_c=1, w_y=1, w_yF=1,
                 c_pred_in_logic=False,
                 c_pred_in_rec=False,
                 orig_rule_sym_to_name=None,
                 reset_selector=True, reset_selector_every_n_epochs=30,
                 intervene=False,
                 mutex=False):
        """
        Args:
            encoder: Torch module with forward method that returns (c_probs, emb)
            emb_size: Embedding size
            rule_emb_size: Rule embedding size
            n_tasks: Number of tasks
            n_rules: Allowed number of rules per task
            n_concepts: Number of concepts
            concept_names: Names of the concepts
            rule_module: RuleModule instance
            lr: Learning rate
            selector_input: Type of selector input
            rel_concept_counts: Relative concept counts
            weight_concepts: Whether to weigh the concept reconstruction loss based on relative concept counts
            w_c: Weight for the concept reconstruction loss (w.r.t. the task loss)
            orig_rule_sym_to_name: None if printing rules should show pos polarity and irrelevance, otherwise prints pos and neg polarity
        """
        super().__init__()

        assert not mutex

        self.save_hyperparameters(ignore=["encoder", "rule_module"])

        self.reset_selector = reset_selector
        self.reset_selector_every_n_epochs = reset_selector_every_n_epochs

        self.lr = lr
        self.embedding_size = emb_size
        self.rule_emb_size = rule_emb_size
        self.n_tasks = n_tasks
        self.n_concepts = n_concepts
        self.n_rules = n_rules
        self.effective_n_rules = n_rules
        self.selector_input = selector_input
        self.rel_concept_counts = rel_concept_counts
        self.weight_concepts = weight_concepts
        self.rule_logger = None
        self.concept_names = concept_names
        self.w_c = w_c
        self.w_y = w_y
        self.w_yF = w_yF
        self.orig_rule_sym_to_name = orig_rule_sym_to_name
        self.skip_info = False
        self.c_pred_in_logic = c_pred_in_logic
        self.c_pred_in_rec = c_pred_in_rec
        self.freeze_rules = False
        self.intervene = intervene

        self.encoder = encoder

        self.rule_module = rule_module(self.rule_emb_size, self.n_concepts, self.n_tasks, self.n_rules)

        self.info = defaultdict(list)
        self.val_info = defaultdict(list)

        self.added_rules = []  # list of lists of rules -- luon rong, khong port RuleAdd intervention (xem docstring file)

        self.rule_mask = torch.ones(self.n_tasks, self.n_rules)

        self.initialized = False
        self.initialize_rule_selector(selector_input)

    def initialize_rule_selector(self, selector_input):
        if self.initialized:
            def initialize_weights(module):
                if isinstance(module, torch.nn.Linear):
                    module.reset_parameters()
            self.neural_rule_selector.apply(initialize_weights)
            return
        self.initialized = True

        if selector_input == InputTypes.concepts or selector_input == InputTypes.concepts_ground_truth:
            selector_input_size = self.n_concepts
        elif selector_input == InputTypes.embedding:
            selector_input_size = self.embedding_size
        else:
            raise NotImplementedError
        self.selector_input_size = selector_input_size
        self.selector_input = selector_input

        self.neural_rule_selector = torch.nn.Sequential(
            torch.nn.Linear(selector_input_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.n_tasks * self.effective_n_rules),
        ).to(self.device)

        if self.effective_n_rules > self.rule_mask.shape[1]:
            self.rule_mask = torch.cat([self.rule_mask, torch.ones(self.n_tasks, self.effective_n_rules - self.rule_mask.shape[1])], dim=-1)

    def decode_rules(self, rule_embs):
        decoded_rules = self.rule_module.decode_rules(rule_embs)  # tasks, rules, concepts, 3
        if not self.training:  # enforce crisp rules
            d_flat = decoded_rules.view(-1, 3)
            max_indices_flat = torch.argmax(d_flat, dim=-1)
            temp = torch.zeros_like(d_flat)
            temp[torch.arange(d_flat.size(0)), max_indices_flat] = 1
            decoded_rules = temp.view(decoded_rules.shape)
        if self.freeze_rules:
            decoded_rules = decoded_rules.detach()
        return decoded_rules

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def get_selector_input(self, batch_c, emb, _old1, c_pred, _old2):
        if self.intervene:
            return batch_c
        if self.selector_input == InputTypes.concepts:
            return c_pred.detach()
        elif self.selector_input == InputTypes.embedding:
            return emb
        elif self.selector_input == InputTypes.concepts_ground_truth:
            return batch_c
        else:
            raise NotImplementedError

    def get_all_rule_vars(self):
        """Returns all rule (pos pol, neg pol, irr) values."""
        r = self.rule_module.rules.weight
        r = r.view(self.n_tasks, self.n_rules, self.rule_emb_size)
        rules = self.decode_rules(r)  # tasks, rules, concepts, 3
        return rules

    def forward(self, x):
        batch_x, batch_c, batch_y = x
        batch_size = batch_c.shape[0]

        c_probs, emb = self.encoder(batch_x)
        c_pred = c_probs

        r = self.get_all_rule_vars()

        selector_input = self.get_selector_input(batch_c, emb, None, c_pred, batch_y)
        logits_s = self.neural_rule_selector(selector_input).view(-1, self.n_tasks, self.effective_n_rules)
        log_p_s = torch.log_softmax(logits_s, dim=-1)  # batch, task, rules
        p_s = torch.softmax(logits_s, dim=-1)

        y_to_mask = torch.ones_like(batch_y).unsqueeze(2).repeat(1, 1, self.effective_n_rules)

        entr = torch.sum(batch_y * torch.sum(-p_s * torch.log(p_s + EPS), dim=-1)) / batch_y.shape[0]

        pospolarity = r[:, :, :, 0]  # task, rule, concept
        irrelevance = r[:, :, :, 2]
        relevance = 1 - irrelevance

        c_intv = batch_c.clone()
        batch_c = batch_c.unsqueeze(1).unsqueeze(1).repeat(1, self.n_tasks, self.effective_n_rules, 1)
        _pospolarity = pospolarity.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        _relevance = relevance.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        p_c_rec = self.rule_module.calc_c_rec(_pospolarity, _relevance)
        if self.training and not self.c_pred_in_logic:
            y_per_rule = self.rule_module.calc_y(batch_c, _pospolarity, _relevance)
        elif self.intervene:
            y_per_rule = self.rule_module.calc_y(batch_c, _pospolarity, _relevance)
            c_pred = c_intv
        else:  # use thresholded c_pred for y_pred (CMR: luon vao nhanh nay, xem docstring)
            y_per_rule = self.rule_module.calc_y((c_pred.detach() > 0.5).float().unsqueeze(1).unsqueeze(1).repeat(1, self.n_tasks, self.effective_n_rules, 1), _pospolarity, _relevance)

        return log_p_s, p_c_rec, y_per_rule, c_pred, p_s, entr, y_to_mask

    def predict_proba(self, x):
        """Xac suat lien tuc theo task (khac predict() ban goc -- ho tra ve
        boolean threshold 0.5 doc lap tung task, khong dam bao 1 nhan duy
        nhat cho bai toan single-label cua ta -- xem baseline_cmr.py)."""
        _, _, y_per_rule, c_pred, p_s, _, _ = self.forward(x)
        return torch.einsum("btr,btr->bt", p_s, y_per_rule), c_pred

    def predict(self, x):
        _, _, y_per_rule, c_pred, p_s, _, _ = self.forward(x)
        y_pred = torch.einsum("btr,btr->bt", p_s, y_per_rule)
        y_pred = y_pred > 0.5
        return y_pred

    def predict_concepts(self, x):
        _, _, _, c_pred, _, _, _ = self.forward(x)
        return c_pred > 0.5

    def training_step(self, batch, batch_idx):
        batch_x, batch_c, batch_y = batch

        (log_p_s, p_c_rec, p_y, p_c, p_s, entr, y_to_mask) = self.forward(batch)

        b_y_btr = batch_y.unsqueeze(2).repeat(1, 1, self.effective_n_rules)
        b_c_btrc = batch_c.unsqueeze(1).unsqueeze(2).repeat(1, self.n_tasks, self.effective_n_rules, 1)
        b_y_btrc = b_y_btr.unsqueeze(3).repeat(1, 1, 1, self.n_concepts)

        true_log_p_y = -binary_cross_entropy(p_y, b_y_btr, reduction="none")
        if not self.c_pred_in_rec:
            true_log_p_c_rec = -binary_cross_entropy(p_c_rec, b_c_btrc, reduction="none")
        else:
            c_pred_btrc = (p_c.detach() > 0.5).float().unsqueeze(1).unsqueeze(2).repeat(1, self.n_tasks, self.effective_n_rules, 1)
            true_log_p_c_rec = -binary_cross_entropy(p_c_rec, c_pred_btrc, reduction="none")
        true_log_p_c = -binary_cross_entropy(p_c, batch_c, reduction="none")

        if not self.weight_concepts:
            sum1 = torch.sum(b_y_btrc * true_log_p_c_rec, dim=-1)
        else:
            concept_w = 1 / self.rel_concept_counts.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_y.shape[0], self.n_tasks, self.effective_n_rules, 1).to(self.device)
            i_concept_w = 1 / (1 - self.rel_concept_counts.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_y.shape[0], self.n_tasks, self.effective_n_rules, 1)).to(self.device)
            w = torch.where(b_c_btrc == 1, concept_w, i_concept_w)
            sum1 = torch.sum(b_y_btrc * w * true_log_p_c_rec, dim=-1)

        w2 = torch.ones_like(batch_y.unsqueeze(-1).repeat(1, 1, self.effective_n_rules))
        w2[batch_y == 0] = self.w_yF
        temp = log_p_s + 0.1 * self.w_y * w2 * true_log_p_y + self.w_c * sum1

        temp = torch.logsumexp(temp, dim=-1)

        logprob_per_sample = torch.sum(temp, dim=-1)
        logprob_per_sample = logprob_per_sample + torch.sum(true_log_p_c, dim=-1)

        loss = torch.mean(-logprob_per_sample)

        if self.skip_info:
            return loss

        p_s = torch.exp(log_p_s)

        y_pred = torch.einsum("btr,btr->bt", p_s[y_to_mask[:, 0, 0] == 1], p_y[y_to_mask[:, 0, 0] == 1])
        avg_c_per_task = torch.einsum("br,brc->bc", p_s[torch.logical_and(batch_y == 1, y_to_mask[:, :, 0] == 1)],
                                       p_c_rec[torch.logical_and(batch_y == 1, y_to_mask[:, :, 0] == 1)])
        c_true = batch_c.unsqueeze(1).repeat(1, self.n_tasks, 1)
        c_true = c_true[torch.logical_and(batch_y == 1, y_to_mask[:, :, 0] == 1)]
        c_true = c_true.view(-1, self.n_concepts).detach().cpu().numpy()
        c_avg_pred = avg_c_per_task.detach().cpu().reshape(-1, self.n_concepts).numpy().round()
        c_true, c_avg_pred = c_true.flatten(), c_avg_pred.flatten()
        c_avg_accuracy = accuracy_score(c_true, c_avg_pred)
        c_accuracy = accuracy_score(batch_c.cpu().numpy().flatten(), p_c.detach().cpu().numpy().round().flatten())
        y_pred = y_pred > 0.5
        y_accuracy_logic = accuracy_score(batch_y.detach().cpu()[y_to_mask[:, 0, 0].to("cpu") == 1], y_pred.detach().cpu())

        self.info["c_accuracy"].append(c_accuracy)
        self.info["c'_accuracy"].append(c_avg_accuracy)
        self.info["y_accuracy"].append(y_accuracy_logic)
        self.info["entropy"].append(entr.item())
        self.info["loss"].append(loss.item())
        self.info["p_s"].append(p_s.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        self.skip_info = True
        val_loss = self.training_step(batch, batch_idx)
        self.val_info["loss"].append(val_loss.item())
        self.skip_info = False
        y_pred = self.predict(batch)
        y_acc = accuracy_score(y_pred.detach().cpu() > 0.5, batch[2].detach().cpu())
        self.val_info["y_accuracy"].append(y_acc)

    def on_validation_epoch_start(self):
        self.val_info = defaultdict(list)

    def on_validation_epoch_end(self):
        self.log("val_loss", sum(self.val_info["loss"]) / len(self.val_info["loss"]))
        self.log("val_acc", -sum(self.val_info["y_accuracy"]) / len(self.val_info["y_accuracy"]))

    def on_train_epoch_start(self):
        self.info = defaultdict(list)
        if self.reset_selector and self.current_epoch % self.reset_selector_every_n_epochs == 0 and self.current_epoch > 0:
            self.initialize_rule_selector(self.selector_input)

    def on_train_epoch_end(self):
        self.log("train_loss", sum(self.info["loss"]) / len(self.info["loss"]))

    def get_rules_sym(self, rule_vars, rule_idx=None, task_idx=None):
        def to_rule_sym(r_idx, t_idx, rule_vars):
            c_type = torch.argmax(rule_vars[t_idx, r_idx, :, :].detach(), dim=-1)
            f = lambda argmax: 1 if argmax == 0 else 0 if argmax == 1 else 9
            r = [f(c_type[k]) for k in range(len(c_type))]
            if self.orig_rule_sym_to_name is None:
                r = [f"({self.concept_names[k]})" if r[k] == 9 else self.concept_names[k] for k in range(len(r)) if r[k] in (1, 9)]
            else:
                r = [f"~{self.concept_names[k]}" if r[k] == 0 else self.concept_names[k] for k in range(len(r)) if r[k] in (1, 0)]
            return " & ".join(r)
        if rule_idx is not None and task_idx is not None:
            return to_rule_sym(rule_idx, task_idx, rule_vars)
        elif rule_idx is not None:
            return [to_rule_sym(rule_idx, t, rule_vars) for t in range(self.n_tasks)]
        elif task_idx is not None:
            return [to_rule_sym(r, task_idx, rule_vars) for r in range(self.effective_n_rules)]
        else:
            return [[to_rule_sym(r, t, rule_vars) for r in range(self.effective_n_rules)] for t in range(self.n_tasks)]

    def aggregate_rules(self, dataloader, type="most_likely", inv=False):
        assert type == "most_likely"  # cac bien the khac (mean_probability/concept_probs) khong dung toi
        rule_vars = self.get_all_rule_vars()
        rules_sym = self.get_rules_sym(rule_vars)
        task_to_rules = {t: {} for t in range(self.n_tasks)}
        task_to_rule_idx = {t: set() for t in range(self.n_tasks)}
        for batch in dataloader:
            log_p_s_x, _, y_per_rule, _, _, _, _ = self(batch)
            rule_idxs = torch.argmax(log_p_s_x, dim=-1)
            for example_idx, y_true in enumerate(batch[2]):
                for task in range(self.n_tasks):
                    if not y_true[task] and not inv:
                        continue
                    if y_true[task] and inv:
                        continue
                    rule_idx = rule_idxs[example_idx, task]
                    rule_sym = rules_sym[task][rule_idx]
                    task_to_rules[task][rule_sym] = task_to_rules[task].get(rule_sym, 0) + 1
                    task_to_rule_idx[task].add(rule_idx.item())
        return task_to_rules, task_to_rule_idx


class CMR(MNISTModel):
    def __init__(self,
                 encoder,
                 emb_size, rule_emb_size,
                 n_tasks, n_rules, n_concepts,
                 concept_names,
                 learning_rate=0.001,
                 selector_input=InputTypes.embedding,
                 reset_selector=True, reset_selector_every_n_epochs=30,
                 rel_concept_counts=None, weight_concepts=False,
                 w_c=1, w_y=1, w_yF=1,
                 c_pred_in_logic=True, c_pred_in_rec=False,
                 orig_rule_sym_to_name=None):
        super().__init__(encoder, emb_size, rule_emb_size, n_tasks, n_rules, n_concepts, concept_names, ProbRDCat,
                          learning_rate, selector_input, rel_concept_counts, weight_concepts, w_c, 10 * w_y, w_yF,
                          c_pred_in_logic, c_pred_in_rec, orig_rule_sym_to_name, reset_selector,
                          reset_selector_every_n_epochs, False, False)
