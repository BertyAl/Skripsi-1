document.addEventListener('DOMContentLoaded', () => {
  // Render flash messages from Flask
  if (window._flash_messages){
    const container = document.createElement('div');
    container.className = 'container mt-3';
    window._flash_messages.forEach(([category, msg]) => {
      const alert = document.createElement('div');
      alert.className = `alert alert-${category} alert-dismissible fade show`;
      alert.innerHTML = `${msg}<button type="button" class="btn-close" data-bs-dismiss="alert"></button>`;
      container.appendChild(alert);
    });
    document.body.prepend(container);
  }
});
