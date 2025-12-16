"""
Celery Application Configuration for Batch Mesh Processing

This module sets up Celery for parallel mesh generation jobs.
Uses Redis as the message broker and result backend.

Setup:
1. Install Redis: brew install redis && brew services start redis
2. Start Celery worker: celery -A celery_app.celery worker --loglevel=info --concurrency=6
"""

from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

# Celery configuration
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Create Celery app
celery = Celery(
    'meshgen',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['tasks']  # Include tasks module
)

# Celery configuration
celery.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Worker settings
    worker_prefetch_multiplier=1,  # Fetch one task at a time for fair distribution
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks (memory leak prevention)
    
    # Result settings
    result_expires=86400,  # Results expire after 24 hours
    
    # Task time limits
    task_soft_time_limit=600,  # 10 minutes soft limit
    task_time_limit=900,       # 15 minutes hard limit
    
    # Task retry settings
    task_acks_late=True,  # Acknowledge task after completion (for reliability)
    task_reject_on_worker_lost=True,
    
    # Concurrency
    worker_concurrency=int(os.environ.get('BATCH_PARALLEL_JOBS', 6)),
)


def get_celery_app():
    """Get the Celery app instance"""
    return celery


if __name__ == '__main__':
    celery.start()
