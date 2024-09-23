css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://media.istockphoto.com/id/1060696342/vector/robot-icon-chat-bot-sign-for-support-service-concept-chatbot-character-flat-style.jpg?s=612x612&w=0&k=20&c=t9PsSDLowOAhfL1v683JMtWRDdF8w5CFsICqQvEvfzY=">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://static.vecteezy.com/system/resources/previews/010/882/225/original/woman-avatar-person-female-illustration-icon-character-face-portrait-woman-avatar-cartoon-girl-user-human-profile-isolated-white-adult-icon-office-headshot-employee-face-head-clipart-vector.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''