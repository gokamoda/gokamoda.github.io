(function () {
  "use strict";

  var containers = document.querySelectorAll(".page__content");

  containers.forEach(function (container, containerIndex) {
    var headings = Array.prototype.slice.call(
      container.querySelectorAll("h2, h3, h4, h5, h6")
    );

    headings.reverse().forEach(function (heading, reverseIndex) {
      var headingLevel = Number(heading.tagName.slice(1));
      var headingIndex = headings.length - reverseIndex - 1;
      var panel = document.createElement("div");
      var button = document.createElement("button");
      var icon = document.createElement("span");
      var panelId = "collapsible-section-" + containerIndex + "-" + headingIndex;
      var sibling = heading.nextSibling;

      panel.className = "collapsible-section__panel";
      panel.id = panelId;

      while (sibling) {
        var nextSibling = sibling.nextSibling;
        var siblingIsHeading =
          sibling.nodeType === 1 && /^H[2-6]$/.test(sibling.tagName);
        var siblingLevel = siblingIsHeading ? Number(sibling.tagName.slice(1)) : 7;

        if (siblingIsHeading && siblingLevel <= headingLevel) {
          break;
        }

        panel.appendChild(sibling);
        sibling = nextSibling;
      }

      icon.className = "collapsible-section__icon";
      icon.setAttribute("aria-hidden", "true");
      icon.textContent = "▼";

      button.className = "collapsible-section__toggle";
      button.type = "button";
      button.setAttribute("aria-label", "Toggle section");
      button.setAttribute("aria-expanded", "true");
      button.setAttribute("aria-controls", panelId);
      button.appendChild(icon);

      heading.classList.add("collapsible-section__heading");
      heading.insertBefore(button, heading.firstChild);
      heading.insertAdjacentElement("afterend", panel);

      button.addEventListener("click", function () {
        var isExpanded = button.getAttribute("aria-expanded") === "true";
        button.setAttribute("aria-expanded", String(!isExpanded));
        panel.hidden = isExpanded;
      });
    });
  });
})();
