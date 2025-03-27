import React, { useState, useRef, useEffect } from 'react';
import './ChatInterface.css';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when messages change
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Call scroll to bottom whenever messages are updated
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    
    if (inputMessage.trim() === '') return;

    // Add user message to chat
    const userMessage = {
      id: Date.now(),
      text: inputMessage,
      sender: 'user'
    };
    setMessages(prevMessages => [...prevMessages, userMessage]);

    const MAX_RETRIES = 10;
    let retryCount = 0;

    const sendMessageWithRetry = async () => {
      try {
        // Send POST request using fetch
        const response = await fetch('http://127.0.0.1:5000/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ message: inputMessage })
        });

        if (!response.ok) {
          throw new Error('Network response was not ok');
        }

        const data = await response.json();
        console.log("I received the response");
        
        const reply = data['response'];
        // Add chatbot response to chat
        const botMessage = {
          id: Date.now() + 1,
          text: reply,
          sender: 'bot'
        };
        setMessages(prevMessages => [...prevMessages, botMessage]);

      } catch (error) {
        console.error(`Attempt ${retryCount + 1} failed:`, error);
        
        if (retryCount < MAX_RETRIES) {
          retryCount++;
          // Exponential backoff: wait increases with each retry
          const waitTime = Math.pow(2, retryCount) * 1000; 
          
          await new Promise(resolve => setTimeout(resolve, waitTime));
          
          // Retry the message send
          await sendMessageWithRetry();
        } else {
          // After 10 attempts, add an error message
          const errorMessage = {
            id: Date.now() + 2,
            text: 'Sorry, there was an error processing your message.',
            sender: 'bot'
          };
          setMessages(prevMessages => [...prevMessages, errorMessage]);
        }
      }
    };

    // Start the send process with retry
    await sendMessageWithRetry();

    // Clear input after sending
    setInputMessage('');
  };

  return (
    <div className="chat-container">
      {/* Chat Messages Container */}
      <div className="chat-messages">
        {messages.map(message => (
          <div 
            key={message.id} 
            className={`message-wrapper ${message.sender}`}
          >
            <div className={`message ${message.sender}`}>
              {message.text}
            </div>
          </div>
        ))}
        {/* Dummy div to enable scrolling to bottom */}
        <div ref={messagesEndRef} />
      </div>

      {/* Message Input */}
      <form 
        onSubmit={handleSendMessage} 
        className="message-input-container"
      >
        <div className="input-wrapper">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Type a message..."
            className="message-input"
          />
          {inputMessage.trim() && (
            <button 
              type="submit" 
              className="send-button"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="send-icon">
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
              </svg>
            </button>
          )}
        </div>
      </form>
    </div>
  );
};

export default ChatInterface;