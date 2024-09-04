from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Chargement du modèle
model = joblib.load(open("D:/ProjetPerso/CreditScoreIA_Banque/App/model.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extraction des valeurs du formulaire
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        
        # Prédiction
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        
        # Affichage du résultat
        return render_template('index.html', prediction_text=f'La réponse à votre demande est : {output}')
    except Exception as e:
        # Gestion des erreurs
        return render_template('index.html', prediction_text='Erreur dans les données fournies.')

if __name__ == "__main__":
    app.run(debug=True)
