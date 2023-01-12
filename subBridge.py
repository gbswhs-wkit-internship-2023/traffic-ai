import requests
import time
import urllib3

urllib3.disable_warnings()
while True:
    url0 = "http://localhost:8000/sub_bridge1"
    url1 = "http://localhost:8000/sub_bridge2"
    url2 = "http://localhost:3000/status"
    url3 = "http://localhost:3000/accidents"

    try:
        res = requests.get(url0, verify=False).json()
        data = {
            'vehicleCount': int(res['vehicleCount']),
            'trafficLight': str(res['trafficLight'])
        }
        res2 = requests.patch(url2, json=data, verify=False).json()
        print(res2)

        res = requests.get(url1, verify=False).json()
        if res['vehicleId'] != -1:
            res2 = requests.post(url3, json=res, verify=False).json()
            print(res2)
    except:
        print('error')
    time.sleep(1)
