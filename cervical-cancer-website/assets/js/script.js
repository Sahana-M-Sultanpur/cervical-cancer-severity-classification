/*
  Main JavaScript file for the CerviClass AI website.
  Each block is commented so beginners can understand what it controls.
*/

const loadingScreen = document.getElementById("loadingScreen");
const menuToggle = document.getElementById("menuToggle");
const navLinks = document.getElementById("navLinks");
const imageUpload = document.getElementById("imageUpload");
const imagePreview = document.getElementById("imagePreview");
const previewWrap = document.getElementById("previewWrap");
const predictionForm = document.getElementById("predictionForm");
const resultCard = document.getElementById("resultCard");
const predictionLabel = document.getElementById("predictionLabel");
const predictionText = document.getElementById("predictionText");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceValue = document.getElementById("confidenceValue");
const contactForm = document.getElementById("contactForm");
const formNote = document.getElementById("formNote");
const heroCanvas = document.getElementById("heroCanvas");
const heroSection = document.getElementById("home");
const heroPanel = document.querySelector(".hero-panel");

// Hide the loading screen after the page finishes loading.
window.addEventListener("load", () => {
  setTimeout(() => {
    loadingScreen.classList.add("hide");
  }, 450);
});

// Open and close the mobile navigation menu.
menuToggle.addEventListener("click", () => {
  const isOpen = navLinks.classList.toggle("open");
  menuToggle.setAttribute("aria-expanded", String(isOpen));
});

// Close the mobile menu when a navigation link is clicked.
document.querySelectorAll(".nav-links a").forEach((link) => {
  link.addEventListener("click", () => {
    navLinks.classList.remove("open");
    menuToggle.setAttribute("aria-expanded", "false");
  });
});

// Animated hero background: floating cytology cells plus AI-style links.
function initHeroAnimation() {
  if (!heroCanvas || !heroSection) {
    return;
  }

  const context = heroCanvas.getContext("2d");
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const pointer = {
    x: 0,
    y: 0,
    active: false
  };

  let width = 0;
  let height = 0;
  let cells = [];
  let animationFrameId = null;

  function createCells() {
    const cellCount = Math.min(38, Math.max(18, Math.floor(width / 52)));
    cells = Array.from({ length: cellCount }, (_, index) => ({
      x: Math.random() * width,
      y: Math.random() * height,
      radius: 16 + Math.random() * 34,
      vx: -0.18 + Math.random() * 0.36,
      vy: -0.12 + Math.random() * 0.24,
      phase: Math.random() * Math.PI * 2,
      nucleusScale: 0.32 + Math.random() * 0.16,
      color: index % 3 === 0 ? "rgba(239, 127, 143, 0.26)" : "rgba(35, 139, 134, 0.18)",
      nucleus: index % 3 === 0 ? "rgba(167, 77, 105, 0.72)" : "rgba(18, 100, 95, 0.72)"
    }));
  }

  function resizeCanvas() {
    const bounds = heroSection.getBoundingClientRect();
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);

    width = Math.max(1, Math.floor(bounds.width));
    height = Math.max(1, Math.floor(bounds.height));
    heroCanvas.width = Math.floor(width * pixelRatio);
    heroCanvas.height = Math.floor(height * pixelRatio);
    heroCanvas.style.width = `${width}px`;
    heroCanvas.style.height = `${height}px`;
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    createCells();
  }

  function drawCell(cell, time) {
    const pulse = Math.sin(time * 0.0014 + cell.phase) * 2.5;
    const distanceToPointer = pointer.active
      ? Math.hypot(cell.x - pointer.x, cell.y - pointer.y)
      : 9999;
    const pointerLift = Math.max(0, 1 - distanceToPointer / 180);
    const radius = cell.radius + pulse + pointerLift * 7;

    context.save();
    context.translate(cell.x, cell.y);
    context.rotate(Math.sin(time * 0.0005 + cell.phase) * 0.22);

    context.beginPath();
    context.ellipse(0, 0, radius * 1.18, radius * 0.86, 0, 0, Math.PI * 2);
    context.fillStyle = cell.color;
    context.fill();
    context.lineWidth = 1.4;
    context.strokeStyle = "rgba(35, 139, 134, 0.18)";
    context.stroke();

    context.beginPath();
    context.ellipse(
      radius * 0.08,
      -radius * 0.02,
      radius * cell.nucleusScale,
      radius * cell.nucleusScale * 0.82,
      0,
      0,
      Math.PI * 2
    );
    context.fillStyle = cell.nucleus;
    context.fill();

    context.restore();
  }

  function drawConnections() {
    context.lineWidth = 1;

    for (let i = 0; i < cells.length; i += 1) {
      for (let j = i + 1; j < cells.length; j += 1) {
        const first = cells[i];
        const second = cells[j];
        const distance = Math.hypot(first.x - second.x, first.y - second.y);

        if (distance < 145) {
          const opacity = 1 - distance / 145;
          context.beginPath();
          context.moveTo(first.x, first.y);
          context.lineTo(second.x, second.y);
          context.strokeStyle = `rgba(35, 139, 134, ${opacity * 0.16})`;
          context.stroke();
        }
      }
    }
  }

  function animate(time = 0) {
    context.clearRect(0, 0, width, height);
    drawConnections();

    cells.forEach((cell) => {
      if (!prefersReducedMotion) {
        cell.x += cell.vx;
        cell.y += cell.vy;

        if (cell.x < -80) cell.x = width + 80;
        if (cell.x > width + 80) cell.x = -80;
        if (cell.y < -80) cell.y = height + 80;
        if (cell.y > height + 80) cell.y = -80;
      }

      drawCell(cell, time);
    });

    if (!prefersReducedMotion) {
      animationFrameId = requestAnimationFrame(animate);
    }
  }

  heroSection.addEventListener("pointermove", (event) => {
    const bounds = heroSection.getBoundingClientRect();
    pointer.x = event.clientX - bounds.left;
    pointer.y = event.clientY - bounds.top;
    pointer.active = true;

    if (heroPanel && window.innerWidth > 980) {
      const rotateY = ((pointer.x / bounds.width) - 0.5) * 8;
      const rotateX = -((pointer.y / bounds.height) - 0.5) * 6;
      heroPanel.style.transform = `perspective(900px) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
    }
  });

  heroSection.addEventListener("pointerleave", () => {
    pointer.active = false;
    if (heroPanel) {
      heroPanel.style.transform = "";
    }
  });

  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();
  animate();

  return () => {
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
    }
  };
}

initHeroAnimation();

// Reveal elements smoothly as they enter the viewport.
const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
      }
    });
  },
  { threshold: 0.14 }
);

document.querySelectorAll(".reveal").forEach((element) => {
  revealObserver.observe(element);
});

// Highlight the active navigation link while scrolling.
const sections = document.querySelectorAll("main section[id]");
const navItems = document.querySelectorAll(".nav-links a");

window.addEventListener("scroll", () => {
  let currentSection = "";

  sections.forEach((section) => {
    const sectionTop = section.offsetTop - 120;
    if (window.scrollY >= sectionTop) {
      currentSection = section.getAttribute("id");
    }
  });

  navItems.forEach((item) => {
    item.classList.toggle("active", item.getAttribute("href") === `#${currentSection}`);
  });
});

// Preview the uploaded image before running the dummy prediction.
imageUpload.addEventListener("change", () => {
  const file = imageUpload.files[0];

  if (!file) {
    previewWrap.hidden = true;
    imagePreview.src = "";
    return;
  }

  imagePreview.src = URL.createObjectURL(file);
  previewWrap.hidden = false;
});

// Dummy labels for the frontend-only prediction demo.
const severityClasses = [
  {
    label: "Superficial-Intermediate",
    message: "The uploaded image is classified as a normal superficial-intermediate cell in this demo result.",
    minConfidence: 82
  },
  {
    label: "Parabasal",
    message: "The demo model suggests a normal parabasal cell pattern.",
    minConfidence: 74
  },
  {
    label: "Koilocytotic",
    message: "The demo model suggests an abnormal koilocytotic cell pattern.",
    minConfidence: 70
  },
  {
    label: "Dyskeratotic",
    message: "The demo model suggests an abnormal dyskeratotic cell pattern.",
    minConfidence: 76
  },
  {
    label: "Metaplastic",
    message: "The demo model suggests a benign metaplastic cell pattern.",
    minConfidence: 78
  }
];

// Generate a dummy prediction when the user clicks the Predict button.
predictionForm.addEventListener("submit", (event) => {
  event.preventDefault();

  if (!imageUpload.files.length) {
    predictionLabel.textContent = "Please upload an image";
    predictionText.textContent = "Choose a Pap smear image first, then click Predict.";
    resultCard.querySelector(".result-status").textContent = "Image required";
    confidenceFill.style.width = "0%";
    confidenceValue.textContent = "0% confidence";
    return;
  }

  resultCard.querySelector(".result-status").textContent = "Analyzing image...";
  predictionLabel.textContent = "Processing";
  predictionText.textContent = "Running dummy frontend prediction...";
  confidenceFill.style.width = "12%";
  confidenceValue.textContent = "Loading";

  // A small delay makes the dummy prediction feel more realistic.
  setTimeout(() => {
    const prediction = severityClasses[Math.floor(Math.random() * severityClasses.length)];
    const confidence = prediction.minConfidence + Math.floor(Math.random() * 15);

    resultCard.querySelector(".result-status").textContent = "Demo prediction ready";
    predictionLabel.textContent = prediction.label;
    predictionText.textContent = prediction.message;
    confidenceFill.style.width = `${confidence}%`;
    confidenceValue.textContent = `${confidence}% confidence`;
  }, 900);
});

// Demo-only contact form response.
contactForm.addEventListener("submit", (event) => {
  event.preventDefault();
  formNote.textContent = "Thank you. This demo form has captured your message locally.";
  contactForm.reset();
});
