import { cn } from "~/lib/ui/utils/utils";

// This className has custom CSS applied to it in our global stylesheet.
const FAKE_H1_CLASS_NAME = "fake-h1";

type FakeH1Props = {
  children: React.ReactNode;
  style?: React.CSSProperties;
  className?: string;
};

/**
 * This is an H2 tag which specific styling overrides to match our H1 tag
 * styles. This is to avoid rendering multiple H1 tags on a single page
 * (which is problematic for HTML semantics/SEO), while allowing us to
 * still have multiple heading titles which appear like H1s.
 */
export function FakeH1({ children, className, style }: FakeH1Props) {
  return (
    <h2 className={cn(FAKE_H1_CLASS_NAME, className)} style={style}>
      {children}
    </h2>
  );
}
