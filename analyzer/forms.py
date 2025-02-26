from django import forms


class SentimentForm(forms.Form):
    text = forms.CharField(label='Enter Text', widget=forms.Textarea)