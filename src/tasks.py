from .celery_app import app

@app.task
def sample_task():
    print("Hello from Celery")