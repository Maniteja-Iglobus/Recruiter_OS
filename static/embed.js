(function () {
  // --- Styling ---
  const styles = `
    #jobsetu-chat-wrapper {
      position: fixed;
      bottom: 20px;
      right: 20px;
      z-index: 999999;
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 15px;
      font-family: 'Inter', sans-serif;
    }
    
    #jobsetu-chat-fab {
      width: 60px;
      height: 60px;
      border-radius: 50%;
      background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
      box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4);
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      color: white;
    }
    
    #jobsetu-chat-fab:hover {
      transform: scale(1.1) rotate(5deg);
      box-shadow: 0 20px 25px -5px rgba(99, 102, 241, 0.5);
    }
    
    #jobsetu-chat-iframe-container {
      width: 380px;
      height: 600px;
      max-height: 80vh;
      background: white;
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
      border: 1px solid rgba(0, 0, 0, 0.05);
      display: none;
      opacity: 0;
      transform: translateY(20px);
      transition: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    #jobsetu-chat-iframe-container.active {
      display: block;
      opacity: 1;
      transform: translateY(0);
    }
    
    @media (max-width: 480px) {
      #jobsetu-chat-iframe-container {
        width: calc(100vw - 40px);
        height: 70vh;
      }
    }
  `;

  const styleSheet = document.createElement("style");
  styleSheet.innerText = styles;
  document.head.appendChild(styleSheet);

  // --- HTML Elements ---
  const wrapper = document.createElement("div");
  wrapper.id = "jobsetu-chat-wrapper";

  const container = document.createElement("div");
  container.id = "jobsetu-chat-iframe-container";

  const iframe = document.createElement("iframe");
  iframe.src = "/static/chat-widget.html";
  iframe.style.width = "100%";
  iframe.style.height = "100%";
  iframe.style.border = "none";

  container.appendChild(iframe);

  const fab = document.createElement("div");
  fab.id = "jobsetu-chat-fab";
  fab.innerHTML = `
    <svg viewBox="0 0 24 24" width="28" height="28" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
    </svg>
  `;

  wrapper.appendChild(container);
  wrapper.appendChild(fab);
  document.body.appendChild(wrapper);

  // --- Interaction ---
  let isOpen = false;
  fab.onclick = () => {
    isOpen = !isOpen;
    if (isOpen) {
      container.classList.add("active");
      fab.innerHTML = `
        <svg viewBox="0 0 24 24" width="28" height="28" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      `;
    } else {
      container.classList.remove("active");
      fab.innerHTML = `
        <svg viewBox="0 0 24 24" width="28" height="28" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
        </svg>
      `;
    }
  };
})();
