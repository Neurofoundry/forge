/* ═══════════════════════════════════════════════════════════════════════════
   NEUROFOUNDRY PAGE TRANSITIONS
   Smooth page transitions with loading states
   ═══════════════════════════════════════════════════════════════════════════ */

(function() {
  'use strict';

  // Create transition overlay
  const overlay = document.createElement('div');
  overlay.className = 'page-transition';
  overlay.innerHTML = '<div class="loader"></div>';
  document.body.appendChild(overlay);

  // Intercept all internal links
  document.addEventListener('click', (e) => {
    const link = e.target.closest('a');
    if (!link) return;
    
    const href = link.getAttribute('href');
    if (!href) return;
    
    // Skip external links, anchors, and special links
    if (href.startsWith('http') || 
        href.startsWith('#') || 
        href.startsWith('mailto:') ||
        href.startsWith('tel:') ||
        link.target === '_blank') {
      return;
    }

    e.preventDefault();
    
    // Show transition
    overlay.classList.add('active');
    
    // Navigate after animation
    setTimeout(() => {
      window.location.href = href;
    }, 400);
  });

  // Hide overlay on page load
  window.addEventListener('load', () => {
    document.body.style.opacity = '0';
    
    requestAnimationFrame(() => {
      document.body.style.transition = 'opacity 0.4s ease';
      document.body.style.opacity = '1';
      overlay.classList.remove('active');
    });
  });

  // Handle browser back/forward
  window.addEventListener('pageshow', (e) => {
    if (e.persisted) {
      overlay.classList.remove('active');
      document.body.style.opacity = '1';
    }
  });

})();
