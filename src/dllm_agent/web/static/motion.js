export function observeReducedMotion(onReduce) {
  const preference = window.matchMedia("(prefers-reduced-motion: reduce)");
  const handleChange = (event) => {
    if (event.matches) onReduce();
  };
  preference.addEventListener("change", handleChange);
  if (preference.matches) onReduce();
}
