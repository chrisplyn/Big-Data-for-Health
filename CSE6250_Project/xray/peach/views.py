from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect
from .forms import UploadFileForm
from .models import Event
from .handleImage import handle_uploaded_file
from django.utils import timezone
from django.urls import reverse
from .xrayConstants import class_names
import numpy as np


def diagnosis_list(request):
	events = Event.objects.filter(pub_date__lte=timezone.now()).order_by('-pub_date')[:10]
	return render(request, 'peach/index.html', {'events': events})

def upload_file(request):
	if request.method == 'POST':
		uploadForm = UploadFileForm(request.POST, request.FILES)
		if uploadForm.is_valid():
			event = Event()
			event.title_text = uploadForm.cleaned_data["title"]
			event.xray_image = uploadForm.cleaned_data["image"]
			event.pub_date = timezone.now()
			event.save()
			prediction_result = handle_uploaded_file(event)
			event_id = event.id
			return HttpResponseRedirect(reverse('peach:detail', args=(event_id,)))
	else:
		uploadForm = UploadFileForm()
	return render(request, 'peach/upload.html', {'form': uploadForm})

def detail(request, event_id):
	event = get_object_or_404(Event, pk=event_id)
	return render(request, 'peach/detail.html', {'event': event})

