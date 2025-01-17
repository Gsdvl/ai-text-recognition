import gradio as gr
import pickle
import os

with open("./models/model.pkl", "rb") as f:
    model = pickle.load(f)

def classify_text(prompt):
    try:
        prediction = model.predict([prompt])[0]
        probability = model.predict_proba([prompt]).max()
        
        # Ajustar para formato do output do modelo
        result = f"Classificação: {'Escrita por LLM' if prediction == 1 else 'Escrita por Humano'}\n"
        result += f"Confiança: {probability:.2%}"
        return result
    except Exception as e:
        return f"Erro ao processar o texto: {str(e)}"
    
interface = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(label="Insira a redação", placeholder="Digite o texto aqui..."),
    outputs=gr.Textbox(label="Resultado da Classificação"),
    title="Classificador de Redações",
    description="Este modelo identifica se uma redação foi escrita por um humano ou por uma LLM.",
)

if __name__ == "__main__":
    interface.launch(share=True, debug=True)
