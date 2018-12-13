from django.utils import timezone
from django.db import models
import datetime
import json

class Event(models.Model):
	title_text = models.CharField(max_length = 200)
	xray_image = models.ImageField(upload_to='peach')
	pub_date = models.DateTimeField('date published')
	pre_result = models.CharField(max_length=200)

	def set_pre_result(self, x):
		self.pre_result = json.dumps(x)
	def get_pre_result(self):
		return json.loads(self.pre_result)

	