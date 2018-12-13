from django import forms

class UploadFileForm(forms.Form):
	title = forms.CharField(max_length = 200)
	image = forms.ImageField()