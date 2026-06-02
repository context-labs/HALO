import { createFileRoute } from "@tanstack/react-router";

import { SettingsPage } from "~/halo/HaloPages";

export const Route = createFileRoute("/settings")({
  component: SettingsPage,
});
