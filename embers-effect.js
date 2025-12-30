// Burning Embers Effect - Reusable across all pages
// Creates floating ember particles from a divider line

(function() {
  // Create ember styles
  const style = document.createElement('style');
  style.textContent = `
    .ember {
      position: absolute;
      width: 4px;
      height: 4px;
      background: radial-gradient(circle, #ff7840 0%, #e0473c 50%, transparent 100%);
      border-radius: 50%;
      box-shadow:
        0 0 8px #e0473c,
        0 0 16px rgba(255, 120, 80, 0.8),
        0 0 24px rgba(255, 100, 60, 0.4);
      pointer-events: none;
      z-index: 100;
      opacity: 0.9;
    }
    
    .spark {
      position: fixed;
      width: 3px;
      height: 8px;
      background: linear-gradient(180deg, #ffffff 0%, #ffd700 30%, #ff8c00 70%, transparent 100%);
      border-radius: 2px;
      box-shadow:
        0 0 6px #ffd700,
        0 0 12px rgba(255, 215, 0, 0.6);
      pointer-events: none;
      z-index: 100;
      opacity: 0.95;
    }
  `;
  document.head.appendChild(style);
  
  // Ember creation function
  function createEmber() {
    const divider = document.querySelector('.section-divider, .hero-section, header');
    if (!divider) return;
    
    const startX = Math.random() * window.innerWidth;
    const rect = divider.getBoundingClientRect();
    const scrollY = window.scrollY || window.pageYOffset;
    const startY = rect.bottom + scrollY;
    
    const ember = document.createElement('span');
    ember.className = 'ember';
    ember.style.left = startX + 'px';
    ember.style.top = startY + 'px';
    
    const riseDistance = 60 + Math.random() * 80;
    const drift = (Math.random() - 0.5) * 40;
    const duration = (2000 + Math.random() * 1500) * 0.9;
    const size = 2 + Math.random() * 3;
    
    ember.style.width = size + 'px';
    ember.style.height = size + 'px';
    ember.style.transition = `transform ${duration}ms ease-out, opacity ${duration}ms ease-out`;
    
    document.body.appendChild(ember);
    
    requestAnimationFrame(() => {
      ember.style.transform = `translate(${drift}px, -${riseDistance}px)`;
      ember.style.opacity = '0';
    });
    
    setTimeout(() => ember.remove(), duration + 50);
  }
  
  // Schedule ember creation
  function scheduleEmbers() {
    createEmber();
    const nextDelay = (80 + Math.random() * 120) * 0.8;
    setTimeout(scheduleEmbers, nextDelay);
  }
  
  // Spark creation function (bottom-right corner)
  function createSpark() {
    const startX = window.innerWidth - 60;
    const startY = window.innerHeight - 60;
    
    const spark = document.createElement('span');
    spark.className = 'spark';
    spark.style.left = startX + 'px';
    spark.style.top = startY + 'px';
    
    const angleVariation = (Math.random() - 0.5) * 60;
    const distance = 40 + Math.random() * 80;
    const radians = (angleVariation - 90) * (Math.PI / 180);
    const driftX = Math.cos(radians) * distance;
    const driftY = Math.sin(radians) * distance;
    const duration = 800 + Math.random() * 600;
    const rotation = (Math.random() - 0.5) * 720;
    
    spark.style.transition = `transform ${duration}ms ease-out, opacity ${duration}ms ease-out`;
    
    document.body.appendChild(spark);
    
    requestAnimationFrame(() => {
      spark.style.transform = `translate(${driftX}px, ${driftY}px) rotate(${rotation}deg)`;
      spark.style.opacity = '0';
    });
    
    setTimeout(() => spark.remove(), duration + 50);
  }
  
  // Schedule spark creation
  function scheduleSparks() {
    if (Math.random() < 0.7) {
      createSpark();
    }
    const nextDelay = 100 + Math.random() * 200;
    setTimeout(scheduleSparks, nextDelay);
  }
  
  // Initialize when window loads
  window.addEventListener('load', () => {
    scheduleEmbers();
    scheduleSparks();
  });
})();
