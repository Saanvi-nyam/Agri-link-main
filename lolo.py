import joblib

# Load the model
model_path = r'C:\saanvi_code\Agri-link-main\sources\NPKModel.pkl'
loaded_model = joblib.load(model_path)

# Check if the model has a predict method and print its type
if hasattr(loaded_model, 'predict'):
    print("Model loaded successfully with a predict method.")
else:
    print("Model does not have a predict method.")
    print(f"Model type: {type(loaded_model)}")