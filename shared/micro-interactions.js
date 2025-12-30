/* ═══════════════════════════════════════════════════════════════════════════
   NEUROFOUNDRY MICRO-INTERACTIONS
   Enhanced hover effects, cursor trails, and subtle animations
   ═══════════════════════════════════════════════════════════════════════════ */

(function() {
  'use strict';

  // ═══════════════════════════════════════════════════════════════════════════
  // MAGNETIC BUTTONS
  // ═══════════════════════════════════════════════════════════════════════════
  const magneticButtons = document.querySelectorAll('.btn-primary, .btn-secondary');
  
  magneticButtons.forEach(btn => {
    btn.addEventListener('mousemove', (e) => {
      const rect = btn.getBoundingClientRect();
      const x = e.clientX - rect.left - rect.width / 2;
      const y = e.clientY - rect.top - rect.height / 2;
      
      btn.style.transform = `translate(${x * 0.2}px, ${y * 0.2}px)`;
    });
    
    btn.addEventListener('mouseleave', () => {
      btn.style.transform = '';
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // TILT CARDS
  // ═══════════════════════════════════════════════════════════════════════════
  const tiltCards = document.querySelectorAll('.card, .project-card, .work-card, .article-card');
  
  tiltCards.forEach(card => {
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      const y = (e.clientY - rect.top) / rect.height;
      
      const tiltX = (y - 0.5) * 10;
      const tiltY = (x - 0.5) * -10;
      
      card.style.transform = `perspective(1000px) rotateX(${tiltX}deg) rotateY(${tiltY}deg) translateY(-4px)`;
    });
    
    card.addEventListener('mouseleave', () => {
      card.style.transform = '';
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // CURSOR GLOW EFFECT
  // ═══════════════════════════════════════════════════════════════════════════
  const cursorGlow = document.createElement('div');
  cursorGlow.className = 'cursor-glow';
  cursorGlow.innerHTML = `
    <style>
      .cursor-glow {
        position: fixed;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(224, 71, 60, 0.08) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
        transform: translate(-50%, -50%);
        transition: opacity 0.3s ease;
        opacity: 0;
      }
      .cursor-glow.active {
        opacity: 1;
      }
    </style>
  `;
  document.body.appendChild(cursorGlow);

  let cursorTimeout;
  document.addEventListener('mousemove', (e) => {
    cursorGlow.style.left = e.clientX + 'px';
    cursorGlow.style.top = e.clientY + 'px';
    cursorGlow.classList.add('active');
    
    clearTimeout(cursorTimeout);
    cursorTimeout = setTimeout(() => {
      cursorGlow.classList.remove('active');
    }, 1000);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // SCROLL REVEAL ANIMATIONS
  // ═══════════════════════════════════════════════════════════════════════════
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('revealed');
        observer.unobserve(entry.target);
      }
    });
  }, observerOptions);

  // Add reveal styles
  const revealStyles = document.createElement('style');
  revealStyles.textContent = `
    .reveal {
      opacity: 0;
      transform: translateY(30px);
      transition: opacity 0.6s ease, transform 0.6s ease;
    }
    .reveal.revealed {
      opacity: 1;
      transform: translateY(0);
    }
    .reveal-scale {
      opacity: 0;
      transform: scale(0.95);
      transition: opacity 0.6s ease, transform 0.6s ease;
    }
    .reveal-scale.revealed {
      opacity: 1;
      transform: scale(1);
    }
    .reveal-left {
      opacity: 0;
      transform: translateX(-30px);
      transition: opacity 0.6s ease, transform 0.6s ease;
    }
    .reveal-left.revealed {
      opacity: 1;
      transform: translateX(0);
    }
    .reveal-right {
      opacity: 0;
      transform: translateX(30px);
      transition: opacity 0.6s ease, transform 0.6s ease;
    }
    .reveal-right.revealed {
      opacity: 1;
      transform: translateX(0);
    }
  `;
  document.head.appendChild(revealStyles);

  // Observe elements
  document.querySelectorAll('.reveal, .reveal-scale, .reveal-left, .reveal-right').forEach(el => {
    observer.observe(el);
  });

  // Auto-add reveal class to common elements
  document.querySelectorAll('.card, .project-card, .article-card, .skill-item, .work-card').forEach((el, i) => {
    el.classList.add('reveal');
    el.style.transitionDelay = `${i * 0.1}s`;
    observer.observe(el);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // RIPPLE EFFECT ON BUTTONS
  // ═══════════════════════════════════════════════════════════════════════════
  const rippleStyles = document.createElement('style');
  rippleStyles.textContent = `
    .ripple {
      position: absolute;
      border-radius: 50%;
      background: rgba(255, 255, 255, 0.3);
      transform: scale(0);
      animation: ripple-animation 0.6s ease-out;
      pointer-events: none;
    }
    @keyframes ripple-animation {
      to {
        transform: scale(4);
        opacity: 0;
      }
    }
  `;
  document.head.appendChild(rippleStyles);

  document.querySelectorAll('.btn, .nav-button, .social-btn').forEach(btn => {
    btn.style.position = 'relative';
    btn.style.overflow = 'hidden';
    
    btn.addEventListener('click', (e) => {
      const ripple = document.createElement('span');
      ripple.className = 'ripple';
      
      const rect = btn.getBoundingClientRect();
      const size = Math.max(rect.width, rect.height);
      
      ripple.style.width = ripple.style.height = size + 'px';
      ripple.style.left = (e.clientX - rect.left - size / 2) + 'px';
      ripple.style.top = (e.clientY - rect.top - size / 2) + 'px';
      
      btn.appendChild(ripple);
      
      setTimeout(() => ripple.remove(), 600);
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // SMOOTH NUMBER COUNTING
  // ═══════════════════════════════════════════════════════════════════════════
  function animateNumber(el, target, duration = 2000) {
    const start = 0;
    const startTime = performance.now();
    
    function update(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Ease out quad
      const eased = 1 - (1 - progress) * (1 - progress);
      const current = Math.floor(start + (target - start) * eased);
      
      el.textContent = current.toLocaleString();
      
      if (progress < 1) {
        requestAnimationFrame(update);
      }
    }
    
    requestAnimationFrame(update);
  }

  // Auto-animate numbers with data-count attribute
  document.querySelectorAll('[data-count]').forEach(el => {
    const target = parseInt(el.dataset.count, 10);
    const observer = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting) {
        animateNumber(el, target);
        observer.disconnect();
      }
    });
    observer.observe(el);
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // TEXT TYPING EFFECT
  // ═══════════════════════════════════════════════════════════════════════════
  window.typeText = function(element, text, speed = 50) {
    return new Promise(resolve => {
      element.textContent = '';
      let i = 0;
      
      function type() {
        if (i < text.length) {
          element.textContent += text.charAt(i);
          i++;
          setTimeout(type, speed);
        } else {
          resolve();
        }
      }
      
      type();
    });
  };

  // ═══════════════════════════════════════════════════════════════════════════
  // PARALLAX SCROLLING
  // ═══════════════════════════════════════════════════════════════════════════
  const parallaxElements = document.querySelectorAll('[data-parallax]');
  
  if (parallaxElements.length) {
    window.addEventListener('scroll', () => {
      const scrollY = window.pageYOffset;
      
      parallaxElements.forEach(el => {
        const speed = parseFloat(el.dataset.parallax) || 0.5;
        const yPos = -(scrollY * speed);
        el.style.transform = `translateY(${yPos}px)`;
      });
    }, { passive: true });
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // NAVBAR SCROLL EFFECT
  // ═══════════════════════════════════════════════════════════════════════════
  const header = document.querySelector('header, .nav-header');
  if (header) {
    let lastScroll = 0;
    
    window.addEventListener('scroll', () => {
      const currentScroll = window.pageYOffset;
      
      if (currentScroll <= 0) {
        header.classList.remove('scroll-up');
        return;
      }
      
      if (currentScroll > lastScroll && !header.classList.contains('scroll-down')) {
        // Scrolling down
        header.classList.remove('scroll-up');
        header.classList.add('scroll-down');
      } else if (currentScroll < lastScroll && header.classList.contains('scroll-down')) {
        // Scrolling up
        header.classList.remove('scroll-down');
        header.classList.add('scroll-up');
      }
      
      lastScroll = currentScroll;
    }, { passive: true });
    
    // Add styles for scroll effect
    const scrollStyles = document.createElement('style');
    scrollStyles.textContent = `
      header, .nav-header {
        transition: transform 0.3s ease;
      }
      header.scroll-down, .nav-header.scroll-down {
        transform: translateY(-100%);
      }
      header.scroll-up, .nav-header.scroll-up {
        transform: translateY(0);
      }
    `;
    document.head.appendChild(scrollStyles);
  }

  console.log('🔥 Neurofoundry Micro-Interactions loaded');
})();
