import asyncio
import datetime
from threading import Thread

from stream.models import InOut, InOutTimestamp


hours = [7, 12, 18]
minutes = [0, 40, 25]


class InoutData:
    def __init__(self, inout):
        # self.in_count = 0
        # self.out_count = 0
        self.inout = inout
        self.current_people_in = 0
        self.time = 0
        self.meal_type = inout.meal_type

        print('start save')
        # self.inout = InOut()
        # self.inout.date = datetime.datetime.today()
        # self.inout.meal_type = meal_type
        # self.inout.total_people_in = 0
        # self.inout.created_at = datetime.datetime.now()
        # self.inout.save()
        print('end save')

    def save_data(self):
        self.inout.total_people_in = self.in_count
        self.inout.save()
        inout_timestamp = InOutTimestamp()
        inout_timestamp.five_minutes = self.time
        inout_timestamp.meal_type = self.meal_type
        inout_timestamp.time = datetime.datetime.now()
        inout_timestamp.people_in = self.in_count
        inout_timestamp.people_out = self.out_count
        inout_timestamp.inout_key = self.inout
        inout_timestamp.created_at = datetime.datetime.now()
        inout_timestamp.save()

    def increase_time(self):
        self.time += 5

    def set_count(self, in_cnt, out_cnt):
        # 유입 인원 - 유출 인원
        self.current_people_in = in_cnt - out_cnt


async def worker(meal_type, cnt1, cnt2):
    print('init')
    inout = InOut()
    inout.date = datetime.datetime.today()
    inout.meal_type = meal_type
    inout.total_people_in = 0
    inout.created_at = datetime.datetime.now()
    inout.save()
    inout_data = InoutData(inout)
    print('init comp')

    # 5분 간격으로 12번 반복 = 60분
    for i in range(12):
        # TODO cnt1, cnt2 고정되서 계속 똑같은값 저장하게 되는 개버그
        inout_data.set_count(cnt1, cnt2)
        inout_data.increase_time()
        inout_data.save_data()
        print('waiting')
        await asyncio.sleep(300)


def worker_sync(meal_type, cnt1, cnt2):
    loop = asyncio.new_event_loop()
    coroutine = worker(meal_type, cnt1, cnt2)
    loop.run_until_complete(coroutine)


def save_inout_data(meal_type, cnt1, cnt2):
    print('92u9428239040943\n3294u90ru\n839ur')
    th1 = Thread(target=worker_sync, args=(meal_type, cnt1, cnt2))
    print('92u9428239040943\n3294u90ru\n839ur')
    th1.start()
    print('92u9428239040943\n3294u90ru\n839ur')

