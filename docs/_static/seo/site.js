(function () {
  "use strict";

  document.addEventListener("click", function (event) {
    var target = event.target.closest("[data-seo-event]");
    if (!target) return;

    var detail = {
      event: target.dataset.seoEvent,
      location: target.dataset.seoLocation || "unspecified",
      language: document.documentElement.lang || "unknown",
      destination: target.href || "",
    };

    window.dispatchEvent(
      new CustomEvent("distilrl:seo-event", { detail: detail })
    );

    window.dataLayer = window.dataLayer || [];
    window.dataLayer.push(Object.assign({ event: detail.event }, detail));
  });
})();
