import pickle

with open("arm_b_results.pkl", "rb") as f:
    output = pickle.load(f)

shap_s1 = output["shap_results"]["S1"]["shap_values"]

print(type(shap_s1))
if isinstance(shap_s1, list):
    print(f"List of {len(shap_s1)} arrays")
    print(f"Each array shape: {shap_s1[0].shape}")
else:
    print(f"Array shape: {shap_s1.shape}")