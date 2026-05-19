/* Touchless website — shared scripts.
 *
 * Currently just the scroll-reveal IntersectionObserver. Loaded by
 * every page; pages that don't tag any elements with .reveal get
 * a no-op.
 *
 * Author: Konstantin Markov
 */
(function setupReveal() {
  const els = document.querySelectorAll(".reveal");
  if (!els.length) return;
  if (typeof IntersectionObserver === "undefined") {
    els.forEach(el => el.classList.add("is-visible"));
    return;
  }
  const io = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        io.unobserve(entry.target);
      }
    }
  }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
  els.forEach(el => io.observe(el));
})();

/* Mobile nav: inject a hamburger button into every page's topnav and
 * wire it to expand/collapse the link list. CSS hides the hamburger on
 * desktop and shows it on narrow widths, so this script can run on
 * every page unconditionally. */
(function setupMobileNav() {
  const nav = document.querySelector("nav.topnav");
  if (!nav) return;
  const links = nav.querySelector(".nav-links");
  if (!links) return;

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "nav-toggle";
  btn.setAttribute("aria-label", "Toggle navigation");
  btn.setAttribute("aria-expanded", "false");
  btn.innerHTML =
    '<span class="nav-toggle-bar"></span>' +
    '<span class="nav-toggle-bar"></span>' +
    '<span class="nav-toggle-bar"></span>';
  nav.appendChild(btn);

  function setOpen(open) {
    nav.classList.toggle("nav-open", open);
    btn.setAttribute("aria-expanded", open ? "true" : "false");
  }

  btn.addEventListener("click", () => setOpen(!nav.classList.contains("nav-open")));

  // Tapping a link inside the dropdown closes it before navigation.
  links.addEventListener("click", (e) => {
    if (e.target.closest(".nav-link")) setOpen(false);
  });

  // Escape closes when the menu is open.
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && nav.classList.contains("nav-open")) setOpen(false);
  });

  // Resizing back to desktop auto-closes the dropdown so it doesn't get
  // stuck open if the user rotates their phone or resizes the window.
  const mq = window.matchMedia("(min-width: 881px)");
  mq.addEventListener?.("change", (e) => { if (e.matches) setOpen(false); });
})();
