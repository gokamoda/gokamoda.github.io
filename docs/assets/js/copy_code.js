document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("pre > code").forEach(code => {
    const btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.textContent = "Copy";
    btn.onclick = () => {
      navigator.clipboard.writeText(code.innerText);
      btn.textContent = "Copied!";
      setTimeout(() => (btn.textContent = "Copy"), 1200);
    };
    code.parentElement.style.position = "relative";
    code.parentElement.appendChild(btn);
  });
});

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".highlighter-rouge").forEach(wrapper => {
    const btn = document.createElement("button");
    btn.className = "code-toggle";
    btn.textContent = "Expand";

    btn.onclick = () => {
      wrapper.classList.toggle("expanded");
      btn.textContent = wrapper.classList.contains("expanded")
        ? "Collapse"
        : "Expand";
    };

    wrapper.appendChild(btn);
  });
});