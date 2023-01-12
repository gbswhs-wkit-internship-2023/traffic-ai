from django.db import models


class InOut(models.Model):
    date = models.DateField()
    meal_type = models.IntegerField()
    total_people_in = models.IntegerField()
    created_at = models.DateTimeField(auto_now=True)

    def get_total_people(self):
        return self.total_people_in

    def get_date(self):
        return self.date


class InOutTimestamp(models.Model):
    five_minutes = models.IntegerField(default=5)
    meal_type = models.IntegerField(default=0)
    people = models.IntegerField(default=0)
    inout_key = models.ForeignKey(to=InOut, related_name='inout_timestamp', on_delete=models.CASCADE, db_column='inout_key')
    created_at = models.DateTimeField(auto_now=True)


class InoutTimeline(models.Model):
    people = models.IntegerField(default=0)
    hour = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now=True)

    def get_hour(self):
        return self.hour

    def get_people(self):
        return self.people
