import { createFileRoute } from "@tanstack/react-router";

import { OnboardingPage } from "~/onboarding/OnboardingPage";

export const Route = createFileRoute("/welcome")({
  component: OnboardingPage,
});
