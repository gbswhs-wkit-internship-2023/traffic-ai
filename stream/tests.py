from django.test import TestCase

from stream.models import InOut, InOutTimestamp, InoutTimeline

import datetime
import random


class InOutClass:
    def __init__(self, date, meal_type, total_people):
        self.date = date
        self.meal_type = meal_type
        self.total_people = total_people

    def get_sql(self):
        inout = InOut()
        inout.date = self.date
        inout.meal_type = self.meal_type
        inout.total_people_in = self.total_people
        inout.save()


class InOutTimestampClass:
    def __init__(self, five_min_term, people, inout_key, meal_type):
        self.five_min_term = five_min_term
        self.people = people
        self.inout_key = inout_key
        self.meal_type = meal_type

    def get_sql(self):
        inout_timestamp = InOutTimestamp()
        inout_timestamp.five_minutes = self.five_min_term
        inout_timestamp.meal_type = self.meal_type
        inout_timestamp.people = self.people
        inout_timestamp.inout_key = self.inout_key
        inout_timestamp.save()


class InOutTimelineClass:
    def __init__(self, people, hour):
        self.people = people
        self.hour = hour

    def get_sql(self):
        inout_timeline = InoutTimeline()
        inout_timeline.people = self.people
        inout_timeline.hour = self.hour
        inout_timeline.save()


if __name__ == "__main__":
    inout_sql = ""
    inout_timestamp_sql = ""
    inout_timeline_sql = ""

    inout_auto_increment = 1

    # 오늘 날짜
    base = datetime.datetime.today()
    # 오늘 날짜부터 7일전 까지의 날짜 리스트
    date_list = [base - datetime.timedelta(days=x) for x in range(7)]

    for date in date_list:
        for meal_type in range(3):
            total_people = random.randrange(180, 220)
            inout_c = InOutClass(date, meal_type, total_people)
            inout_sql += inout_c.get_sql()

            for five_min_term in range(5, 61, 5):
                people = random.randrange(0, 90)
                inout_timestamp_c = InOutTimestampClass(five_min_term, people, inout_auto_increment, meal_type)
                inout_timestamp_sql += inout_timestamp_c.get_sql()

            inout_auto_increment += 1

    for hour in range(0, 24):
        people = random.randrange(0, 90)
        inout_timeline_c = InOutTimelineClass(people, hour)
        inout_timeline_sql += inout_timeline_c.get_sql()

    print(inout_sql)
    print(inout_timestamp_sql)
    print(inout_timeline_sql)

