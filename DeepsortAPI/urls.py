from django.urls import path
from stream import views


urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed1/', views.video_feed1, name='video_feed1'),
    path('sub_bridge1', views.traffic_info),
    path('sub_bridge2', views.traffic_event)
]
