from django.urls import path

from . import views

app_name = 'peach'
urlpatterns = [
	path('', views.upload_file, name='upload'),
	path('list/', views.diagnosis_list, name='list'),
	path('<int:event_id>/', views.detail, name='detail')
]