from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render

from stream.streaming import stream, t


def index(request):
    return render(request, 'index.html')


def video_feed1(request):
    source = 'videos/traffic_25fps.mp4'
    # source = "rtsp://localhost:8554/stream"
    return StreamingHttpResponse(stream(source, is_in=True), content_type='multipart/x-mixed-replace; boundary=frame')


def traffic_info(request):
    vehicle_count = t.get_count()

    return JsonResponse({
        "vehicleCount": vehicle_count,
        "trafficLight": t.get_traffic_light()
    })


def traffic_event(request):
    violation = t.get_violatoin()
    if violation:
        return JsonResponse(violation)
    else:
        return JsonResponse({
                "vehicleId": int(-1),
                "accidentTypeLabel": "데이터 없음"
            })
