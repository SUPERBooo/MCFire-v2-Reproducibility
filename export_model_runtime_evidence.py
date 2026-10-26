from common import *
import warnings, joblib, sklearn

rows=[]
for y in YEARS:
    path=ROOTS[y]/f"data/intermediate/S14_COMPACT_SPATIAL_PHYSICS_FUSION/models/D2_S14_FUSION74_LOGREG_{y}.joblib"
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model=joblib.load(path)
    old_versions=sorted(set(getattr(w.message,"original_sklearn_version","") for w in captured)-{""})
    newmodel=joblib.load(OUT/f"model_details/{y}_FUSION74_model.joblib")
    rows.append(dict(year=y,original_model=relative(path),original_model_sha256=sha(path),original_sklearn_version=";".join(old_versions) if old_versions else "same-version-load-no-warning",current_sklearn_version=sklearn.__version__,recorded_input_features=int(model.named_steps["imputer"].n_features_in_),new_input_features=int(newmodel.named_steps["imputer"].n_features_in_),original_estimator_features=int(model.named_steps["logreg"].coef_.shape[1]),new_estimator_features=int(newmodel.named_steps["logreg"].coef_.shape[1]),original_missing_indicator_columns=model.named_steps["imputer"].indicator_.features_.tolist()))
save(pd.DataFrame(rows),"frozen_model_runtime_evidence.csv")
print(pd.DataFrame(rows)[["year","original_sklearn_version","current_sklearn_version","original_estimator_features"]].to_string(index=False))
