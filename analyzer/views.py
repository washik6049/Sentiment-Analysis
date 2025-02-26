import os
import pickle

from django.conf import settings
from django.shortcuts import render

from .forms import SentimentForm


def home(request):
    sentiment = None
    if request.method == 'POST':
        form = SentimentForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']

            # Define paths
            model_path = os.path.join(settings.BASE_DIR, 'analyzer', 'sentiment_model.pkl')
            vectorizer_path = os.path.join(settings.BASE_DIR, 'analyzer', 'vectorizer.pkl')

            try:
                # Load model
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                # Load vectorizer
                with open(vectorizer_path, 'rb') as f:
                    vectorizer = pickle.load(f)

                # Predict sentiment
                text_vectorized = vectorizer.transform([text])
                sentiment = model.predict(text_vectorized)[0]
            except EOFError:
                sentiment = "Error: Model or vectorizer file is empty or corrupted."
            except FileNotFoundError:
                sentiment = "Error: Model or vectorizer file not found."
            except Exception as e:
                sentiment = f"Error: {str(e)}"
    else:
        form = SentimentForm()

    return render(request, 'analyzer/home.html', {'form': form, 'sentiment': sentiment})