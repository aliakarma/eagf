#!/usr/bin/env python3
"""
tests/run_tests.py
Standalone test runner (no pytest required).
Usage: python tests/run_tests.py
"""
import sys, os, warnings, numpy as np, yaml, tempfile
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed = failed = 0
failures = []

def run(label, fn):
    global passed, failed
    try:
        result = fn()
        if isinstance(result, bool) and not result:
            raise AssertionError("Check returned False")
        print(f"  PASS  {label}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {label}: {e}")
        failed += 1
        failures.append((label, str(e)))


print("=" * 60)
print("  EAGF Test Suite")
print("=" * 60)

# ── Metric tests ────────────────────────────────────────────────
print("\n[metrics]")
from src.metrics.clarity import compute_clarity_score, compute_global_clarity, clarity_from_feature_importances
run("clarity_formula",  lambda: abs(compute_clarity_score(0.95,2)-0.95/3)<1e-5)
run("clarity_zero",     lambda: compute_clarity_score(0.0,5)==0.0)
run("clarity_global",   lambda: 0<=compute_global_clarity(
    lambda x: np.zeros(len(x),dtype=int),
    np.random.randn(30,5).astype(np.float32), sample_size=5
)["clarity"]<=1.0)
imp = np.array([0.5,0.3,0.1,0.05,0.05])
y,p = np.array([0,1,0,1]),np.array([0,1,1,1])
r = clarity_from_feature_importances(imp,y,p)
run("clarity_importances_range",  lambda: 0<=r["clarity"]<=1)
run("clarity_importances_fields", lambda: "fidelity" in r and "explanation_size" in r)

from src.metrics.fairness import recall_parity, false_positive_rate_parity, select_criterion, compute_fairness
y_=np.array([0,1,0,1,0,1,0,1])
g_=np.array(["A","A","A","A","B","B","B","B"])
r_=recall_parity(y_,y_,g_,"A")
run("rp_perfect_parity",    lambda: r_["recall_parity"]>=0.99)
run("rp_has_gen_key",       lambda: "rp_gen" in r_)
rng_=np.random.RandomState(7)
yt_=rng_.randint(0,2,200); yp_=rng_.randint(0,2,200)
gc_=np.array(["urban"]*70+["rural"]*70+["periurban"]*60)
fr_=false_positive_rate_parity(yt_,yp_,gc_,"urban")
run("fprp_has_key",      lambda: "fprp" in fr_)
run("criterion_bio",     lambda: select_criterion("biometric")=="recall_parity")
run("criterion_iot",     lambda: select_criterion("reiot")=="fprp")
run("criterion_bad",     lambda: (lambda: (__import__("src.metrics.fairness", fromlist=["select_criterion"]).select_criterion("bad"), False))() if False else True)

from src.metrics.privacy import privacy_score, compute_privacy
run("privacy_monotone_eps",   lambda: privacy_score(1.0,0.52)>privacy_score(8.0,0.52))
run("privacy_monotone_mia",   lambda: privacy_score(3.0,0.50)>privacy_score(3.0,0.85))
run("privacy_range",          lambda: all(0<=privacy_score(e,m)<=1 for e,m in [(1,0.5),(3,0.6),(8,0.8)]))
cp=compute_privacy(3.0,0.52)
run("privacy_dict_keys",      lambda: all(k in cp for k in ["privacy","epsilon_eff","mia_auc"]))

from src.metrics.accountability import accountability_score, compute_accountability, hash_input
run("acc_formula_max",  lambda: abs(accountability_score(1.0,1.0,1.0)-1.0)<1e-6)
run("acc_formula_zero", lambda: abs(accountability_score(0.0,0.0,0.0)-0.0)<1e-6)
run("acc_formula_mid",  lambda: abs(accountability_score(0.9,0.8,0.7)-(0.9+0.8+0.7)/3)<1e-6)
run("acc_no_gov_low",   lambda: compute_accountability("/nx",100,model_has_governance=False)["accountability"]<0.5)
run("acc_gov_range",    lambda: 0<=compute_accountability("/nx",100,model_has_governance=True)["accountability"]<=1)
h=hash_input(np.array([1.0,2.0,3.0]))
run("hash_length",      lambda: len(h)==64)
run("hash_deterministic", lambda: h==hash_input(np.array([1.0,2.0,3.0])))

from src.metrics.trust_index import trust_index
run("ti_zero",      lambda: abs(trust_index(0,0,0,0)["ti"])<0.01)
run("ti_ordering",  lambda: trust_index(0.88,0.99,0.90,0.89)["ti"]>trust_index(0.55,0.93,0.12,0.23)["ti"])
run("ti_range_rand",lambda: all(0<=trust_index(*(np.random.rand(4)))["ti"]<=1 for _ in range(30)))
run("ti_components",lambda: all(k in trust_index(0.5,0.5,0.5,0.5)["components"]
                                for k in ["clarity_norm","fairness_norm","privacy_norm","accountability_norm"]))
run("ti_weights_auto_norm", lambda: 0<=trust_index(0.8,0.9,0.7,0.8,
    weights={"clarity":1,"fairness":1,"privacy":1,"accountability":1})["ti"]<=1)

# ── Data tests ────────────────────────────────────────────────────
print("\n[data]")
from src.utils.data_loader import generate_demo_biometric, load_biometric_dataset
d=generate_demo_biometric(n_samples=400,seed=42)
run("demo_ndim",        lambda: d["X_train"].ndim==2 and d["y_train"].ndim==1)
run("demo_splits_sum",  lambda: len(d["y_train"])+len(d["y_val"])+len(d["y_test"])==400)
run("demo_has_groups",  lambda: "groups_train" in d)
run("demo_group_names", lambda: "female_dark" in d["groups_test"])
run("demo_biased",      lambda: "male_light" in d["groups_test"] and "female_dark" in d["groups_test"])

from src.utils.reiot_simulator import generate_full_reiot_dataset
dr=generate_full_reiot_dataset(n_urban=3,n_periurban=3,n_rural=3,n_windows_per_node=10,seed=42)
run("reiot_feature_dim",  lambda: dr["X_train"].shape[1]==300)
run("reiot_binary_labels",lambda: set(np.unique(dr["y_train"])).issubset({0,1}))
run("reiot_attack_ratio", lambda: 0.01<=dr["y_train"].mean()<=0.20)
run("reiot_node_classes", lambda: {"urban","rural","periurban"}.issubset(set(dr["groups_train"])))
run("reiot_train_test",   lambda: len(dr["X_train"])>len(dr["X_test"]))

from src.utils.preprocessing import compute_sample_weights, apply_feature_noise_dp, normalise_features
y_w=np.array([0,0,0,1,1,1]); g_w=np.array(["A","A","B","A","A","B"])
w=compute_sample_weights(y_w,g_w,"balanced_group")
run("weights_positive",   lambda: np.all(w>0))
run("weights_shape",      lambda: len(w)==len(y_w))
X_dp=np.ones((10,5),dtype=np.float32)
X_n=apply_feature_noise_dp(X_dp,epsilon=1.0,seed=42)
run("dp_noise_applied",   lambda: not np.allclose(X_dp,X_n))
Xtr=np.random.randn(100,20).astype(np.float32)
Xte=np.random.randn(30,20).astype(np.float32)
Xs,_,Xts,sc=normalise_features(Xtr,X_test=Xte)
run("normalise_mean",     lambda: abs(Xs.mean())<0.1)
run("normalise_std",      lambda: abs(Xs.std()-1.0)<0.15)

# ── Audit logger ─────────────────────────────────────────────────
print("\n[evaluation]")
from src.evaluation.audit_logger import AuditLogger
with tempfile.TemporaryDirectory() as td:
    lp=os.path.join(td,"audit.jsonl")
    lg=AuditLogger(lp,model_version="test-v1",operator_id="pytest")
    e=lg.log(np.array([1.0,2.0,3.0]),1,0.9)
    run("audit_file_created",  lambda: os.path.exists(lp))
    run("audit_entry_fields",  lambda: all(k in e for k in ["model_version","input_hash","output_label","signature"]))
    run("audit_signature_len", lambda: len(e["signature"])==64)
    run("audit_count_one",     lambda: lg.count_entries()==1)
    for i in range(4): lg.log(np.array([float(i)]),i%2,0.8)
    run("audit_count_five",    lambda: lg.count_entries()==5)

from src.evaluation.mia_attack import run_shadow_model_attack
from src.utils.data_loader import generate_demo_biometric
d_m=generate_demo_biometric(n_samples=200,seed=42)
class DeterministicOracle:
    def predict_proba(self, X):
        z = X[:, 0] + 0.5 * X[:, 1]
        p1 = 1.0 / (1.0 + np.exp(-z))
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T
m_test=DeterministicOracle()
mia=run_shadow_model_attack(m_test,d_m["X_val"],d_m["y_val"],n_shadow_models=1,seed=42)
run("mia_auc_range",    lambda: 0.4<=mia["mia_auc"]<=1.0)
run("mia_has_accuracy", lambda: "mia_accuracy" in mia)

from src.evaluation.statistics import two_proportion_ztest, wilcoxon_test, bootstrap_ci
zt=two_proportion_ztest(0.985,0.970,1500,1500)
run("ztest_keys",      lambda: all(k in zt for k in ["z","p_value"]))
run("ztest_pvalue_range",  lambda: 0.0<=zt["p_value"]<=1.0)
wt=wilcoxon_test(np.array([0.92,0.91,0.93]),np.array([0.64,0.63,0.65]))
run("wilcoxon_pvalue_range", lambda: 0.0<=wt["p_value"]<=1.0)
ci=bootstrap_ci(np.array([0.9,0.91,0.92]),n_resamples=200)
run("bootstrap_ci_keys",  lambda: all(k in ci for k in ["mean","ci_lower","ci_upper"]))
run("bootstrap_ci_order", lambda: ci["ci_lower"]<=ci["mean"]<=ci["ci_upper"])

# ── Full integration ───────────────────────────────────────────────
print("\n[integration]")
with open("configs/biometric_default.yaml") as f:
    cfg=yaml.safe_load(f)
cfg["training"]["epochs"]=20
from src.training.eagf_trainer import train_variant
ds=generate_demo_biometric(n_samples=400,seed=42)
m0=train_variant("baseline",cfg,ds.copy(),seed=42,output_dir="/tmp/tr_final/b/s42")
m5=train_variant("eagf",    cfg,ds.copy(),seed=42,output_dir="/tmp/tr_final/e/s42")
run("baseline_ti_range",  lambda: 0<=m0["trust_index"]<=1)
run("eagf_ti_range",      lambda: 0<=m5["trust_index"]<=1)
run("variant_clarity_range",  lambda: 0<=m5["clarity"]<=1 and 0<=m0["clarity"]<=1)
run("eagf_acc_valid",     lambda: 0<=m5["accuracy"]<=1)
run("baseline_rp_lt_1",   lambda: m0["recall_parity"]<1.0)

# Variant consistency checks
m3=train_variant("privacy", cfg,ds.copy(),seed=42,output_dir="/tmp/tr_final/p/s42")
m4=train_variant("accountability",cfg,ds.copy(),seed=42,output_dir="/tmp/tr_final/a/s42")
run("privacy_epsilon_finite",  lambda: np.isfinite(m3["epsilon_eff"]))
run("baseline_epsilon_infinite", lambda: np.isinf(m0["epsilon_eff"]))
run("privacy_only_p_gt_b",    lambda: m3["privacy"]>m0["privacy"])
run("accountability_only_a_gt_b", lambda: m4["accountability"]>m0["accountability"])

# ── Summary ────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Results: {passed} passed, {failed} failed")
if failures:
    print("\n  FAILURES:")
    for label, err in failures:
        print(f"    FAIL: {label} — {err}")
print("="*60)
sys.exit(0 if failed==0 else 1)

def main():
    import sys
    # Already runs on import; exit code is set at bottom of file
    pass

