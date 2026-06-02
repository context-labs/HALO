export function GradientCard() {
  return (
    <div
      className={`
        pointer-events-none absolute inset-0 bg-linear-to-br from-stone-500/5
        via-transparent to-slate-500/5
      `}
    />
  );
}
