# import joblib
# import pandas as pd
# import gradio as gr

# # Load trained model
# MODEL_PATH = "model.joblib"
# model = joblib.load(MODEL_PATH)

# # IMPORTANT:
# # The model expects the same one-hot encoded feature set as training.
# # For the exam, we keep this simple and assume the model was trained
# # on numeric + boolean features already aligned.

# def predict(
#     region,
#     primary_role,
#     alignment,
#     status,
#     species,
#     honour,
#     ruthlessness,
#     intelligence,
#     combat,
#     diplomacy,
#     leadership,
#     trait_loyal,
#     trait_scheming,
# ):
#     # Minimal input mapping (exam-sufficient)
#     data = {
#         "honour_1to5": honour,
#         "ruthlessness_1to5": ruthlessness,
#         "intelligence_1to5": intelligence,
#         "combat_skill_1to5": combat,
#         "diplomacy_1to5": diplomacy,
#         "leadership_1to5": leadership,
#         "trait_loyal": int(trait_loyal),
#         "trait_scheming": int(trait_scheming),
#     }

#     df = pd.DataFrame([data])
#     prediction = model.predict(df)[0]

#     return prediction


# iface = gr.Interface(
#     fn=predict,
#     inputs=[
#         gr.Textbox(label="Region"),
#         gr.Textbox(label="Primary Role"),
#         gr.Textbox(label="Alignment"),
#         gr.Textbox(label="Status"),
#         gr.Textbox(label="Species"),
#         gr.Slider(1, 5, value=3, label="Honour (1–5)"),
#         gr.Slider(1, 5, value=3, label="Ruthlessness (1–5)"),
#         gr.Slider(1, 5, value=3, label="Intelligence (1–5)"),
#         gr.Slider(1, 5, value=3, label="Combat Skill (1–5)"),
#         gr.Slider(1, 5, value=3, label="Diplomacy (1–5)"),
#         gr.Slider(1, 5, value=3, label="Leadership (1–5)"),
#         gr.Checkbox(label="Trait: Loyal"),
#         gr.Checkbox(label="Trait: Scheming"),
#     ],
#     outputs=gr.Textbox(label="Predicted House"),
#     title="Game of Thrones House Predictor",
#     description="Decision Tree model trained using Azure ML pipeline",
# )

# if __name__ == "__main__":
#     iface.launch()
