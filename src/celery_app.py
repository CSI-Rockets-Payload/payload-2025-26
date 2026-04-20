# celery.py
from celery import Celery

app = Celery(
    "worker",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0"
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

app.autodiscover_tasks(['src'])

# app = Celery(
#     "payload-2025-26",
#     broker="redis://redis:6379/0",
#     backend="redis://redis:6379/0",
#     include=["src.tasks"], 
# )

# Optional beat schedule
app.conf.beat_schedule = {
    "smaple-task": {
        "task": "src.tasks.sample_task",
        "schedule": 5.0,
        "args": (),
    },

    #schedule more tasks here
}
