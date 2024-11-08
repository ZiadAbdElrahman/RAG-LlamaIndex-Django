from django.urls import path
from .views import chat_and_upload_view, ai_response, upload_file

urlpatterns = [
    path('chatbot/', chat_and_upload_view, name='chat_and_upload'),
    path('ai_response/', ai_response, name='ai_response'),  # New endpoint for AI responses
    path('upload_file/', upload_file, name='upload_pdf'),  # New endpoint for AI responses
]