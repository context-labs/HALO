import type { ComponentType, ReactNode } from "react";
import { useMemo, useState } from "react";
import {
  Activity,
  BadgeCheck,
  Boxes,
  CalendarDays,
  CheckCircle2,
  Code2,
  Database,
  FileJson,
  Grid2X2,
  Info,
  Layers3,
  ListChecks,
  MessageSquareWarning,
  MoreHorizontal,
  PanelRightOpen,
  Play,
  Search,
  Settings2,
  ShieldAlert,
  SlidersHorizontal,
  SquareCheckBig,
  Table2,
  Terminal,
  ToggleRight,
  Wand2,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  XAxis,
} from "recharts";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
  Alert,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogRoot,
  AlertDialogTitle,
  AlertDialogTrigger,
  AlertInfo,
  AlertWarning,
  Avatar,
  AvatarFallback,
  Badge,
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
  Button,
  Calendar,
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
  Centered,
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  Checkbox,
  Code,
  CodeBlock,
  Col,
  CommandBlock,
  DateTimePicker,
  Dialog,
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
  DropdownMenuTrigger,
  EmptyState,
  FakeH1,
  FeatureCard,
  GradientCard,
  Grid,
  InferenceIcon,
  Input,
  JsonComponent,
  Label,
  LoadingScreen,
  MermaidDiagram,
  NavTab,
  NavTabs,
  PatternAvatarFallback,
  Popover,
  PopoverContent,
  PopoverTrigger,
  Prompt,
  RadioGroup,
  RadioGroupItem,
  ResponsiveRow,
  Row,
  ScaleLoader,
  ScoreBadge,
  SearchInput,
  Select,
  SelectableCard,
  Separator,
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
  Skeleton,
  SkeletonAvatar,
  SkeletonButton,
  SkeletonCard,
  SkeletonText,
  Slider,
  Spinner,
  Switch,
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableFooter,
  TableHead,
  TableHeader,
  TableRow,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  Textarea,
  Tooltip,
  toast,
} from "~/lib/ui";
import type { ChartConfig } from "~/lib/ui";

import { useGallerySearch } from "./search";

export type SectionId =
  | "foundations"
  | "inputs"
  | "overlays"
  | "navigation"
  | "custom";

export type GallerySectionFilter = SectionId | "all";

export type IconComponent = ComponentType<{ className?: string }>;

type GallerySectionConfig = {
  id: SectionId;
  label: string;
  icon: IconComponent;
  summary: string;
  components: string[];
};

export const gallerySections: GallerySectionConfig[] = [
  {
    id: "foundations",
    label: "Foundations",
    icon: Layers3,
    summary: "Core surfaces, status language, motion, and feedback basics.",
    components: [
      "Button",
      "Badge",
      "Alert",
      "AlertInfo",
      "AlertWarning",
      "Avatar",
      "Card",
      "Separator",
      "Skeleton",
      "Spinner",
      "ScaleLoader",
    ],
  },
  {
    id: "inputs",
    label: "Inputs",
    icon: SlidersHorizontal,
    summary: "Form controls for desktop workflows and dense settings views.",
    components: [
      "Input",
      "Textarea",
      "Label",
      "Checkbox",
      "Switch",
      "RadioGroup",
      "Slider",
      "Select",
      "Calendar",
      "DateTimePicker",
      "SearchInput",
    ],
  },
  {
    id: "overlays",
    label: "Overlays",
    icon: PanelRightOpen,
    summary: "Layered controls, confirmation moments, and transient messages.",
    components: [
      "Dialog",
      "AlertDialog",
      "Sheet",
      "Popover",
      "DropdownMenu",
      "Tooltip",
      "Toast",
      "Prompt",
    ],
  },
  {
    id: "navigation",
    label: "Navigation + Data",
    icon: Table2,
    summary: "Navigation, disclosure, tabbing, tabular data, and charts.",
    components: [
      "Accordion",
      "Tabs",
      "NavTabs",
      "Breadcrumb",
      "Table",
      "Chart",
    ],
  },
  {
    id: "custom",
    label: "Custom",
    icon: Wand2,
    summary: "Inference-flavored building blocks and content renderers.",
    components: [
      "EmptyState",
      "FakeH1",
      "FeatureCard",
      "GradientCard",
      "InferenceIcon",
      "JsonComponent",
      "LoadingScreen",
      "MermaidDiagram",
      "PatternAvatarFallback",
      "ScoreBadge",
      "SelectableCard",
      "Code",
      "CodeBlock",
      "CommandBlock",
      "Row",
      "Col",
      "Grid",
      "Centered",
      "ResponsiveRow",
      "ThemeToggle",
      "HomePageBackdrop",
    ],
  },
];

const chartData = [
  { name: "Mon", latency: 94, accuracy: 84 },
  { name: "Tue", latency: 72, accuracy: 88 },
  { name: "Wed", latency: 108, accuracy: 91 },
  { name: "Thu", latency: 61, accuracy: 86 },
  { name: "Fri", latency: 82, accuracy: 94 },
  { name: "Sat", latency: 56, accuracy: 96 },
];

const chartConfig = {
  latency: {
    label: "Latency",
    color: "var(--color-detail-brand)",
  },
  accuracy: {
    label: "Quality",
    color: "var(--color-detail-success)",
  },
} satisfies ChartConfig;

const sampleMermaid = `
flowchart LR
  Copy[Copied UI package] --> Theme[Theme tokens]
  Theme --> Gallery[Gallery surface]
  Gallery --> Verify[Build and desktop smoke]
`;

const sampleCode = `import { Button, Card } from "~/lib/ui";

export function ActionCard() {
  return (
    <Card className="p-4">
      <Button>Run evaluation</Button>
    </Card>
  );
}`;

const featureIcon =
  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='32' height='32' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='black'/%3E%3Cpath d='M8 18h16M8 13h10M8 23h7' stroke='white' stroke-width='2' stroke-linecap='round'/%3E%3C/svg%3E";

export const componentCount = gallerySections.reduce(
  (count, section) => count + section.components.length,
  0,
);

export function GalleryPage({
  sectionId = "all",
}: {
  sectionId?: GallerySectionFilter;
}) {
  const { clearQuery, query } = useGallerySearch();
  const [selectedPackage, setSelectedPackage] = useState("core");
  const [selectedNavTab, setSelectedNavTab] = useState("preview");
  const [dialogOpen, setDialogOpen] = useState(false);
  const [completed, setCompleted] = useState(true);
  const [notifications, setNotifications] = useState(true);
  const [radioValue, setRadioValue] = useState("balanced");
  const [sliderValue, setSliderValue] = useState(64);
  const [selectValue, setSelectValue] = useState("production");
  const [calendarDate, setCalendarDate] = useState<Date | undefined>(
    new Date(),
  );
  const [dateTime, setDateTime] = useState<Date | undefined>(new Date());

  const filteredSections = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();

    return gallerySections.filter((section) => {
      const matchesActive =
        sectionId === "all" || section.id === sectionId;

      if (!matchesActive) {
        return false;
      }

      if (!normalizedQuery) {
        return true;
      }

      return [section.label, section.summary, ...section.components]
        .join(" ")
        .toLowerCase()
        .includes(normalizedQuery);
    });
  }, [sectionId, query]);

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-8 px-5 py-6 sm:px-8">
            <header className="grid gap-6 border-b border-subtle pb-6 lg:grid-cols-[1fr_360px] lg:items-end">
              <div>
                <Badge variant="status-brand" size="sm">
                  Local design system
                </Badge>
                <h1 className="mt-4 max-w-3xl text-5xl">
                  Components with a pulse.
                </h1>
                <p className="mt-4 max-w-2xl text-muted-foreground">
                  A desktop gallery for the copied Tailwind UI library, wired
                  to local theme tokens, fonts, overlays, and live interaction
                  states.
                </p>
              </div>
              <div className="rounded-xl border border-subtle bg-card p-4">
                <p className="text-sm font-semibold">
                  {sectionId === "all"
                    ? `${componentCount} total exports`
                    : gallerySections.find((section) => section.id === sectionId)
                        ?.label}
                </p>
                <p className="mt-1 text-sm text-muted-foreground">
                  {query
                    ? `Filtered by "${query}"`
                    : "Search and route state are now managed by TanStack Start."}
                </p>
              </div>
            </header>

            <Grid className="grid-cols-1 gap-4 md:grid-cols-3">
              <MetricCard icon={Boxes} label="Families" value="5" />
              <MetricCard
                icon={Database}
                label="Source"
                value="inference/apps/web"
              />
              <MetricCard icon={CheckCircle2} label="Mode" value="Local copy" />
            </Grid>

            {filteredSections.length === 0 ? (
              <EmptyState
                action={
                  <Button variant="outline" onClick={clearQuery}>
                    Reset search
                  </Button>
                }
                description="No component names match the current filter."
                icon={Search}
                title="No matches"
              />
            ) : null}

            {filteredSections.map((section) => (
              <GallerySection
                components={section.components}
                icon={section.icon}
                key={section.id}
                summary={section.summary}
                title={section.label}
              >
                {section.id === "foundations" ? <FoundationsDemo /> : null}
                {section.id === "inputs" ? (
                  <InputsDemo
                    calendarDate={calendarDate}
                    completed={completed}
                    dateTime={dateTime}
                    notifications={notifications}
                    radioValue={radioValue}
                    selectValue={selectValue}
                    setCalendarDate={setCalendarDate}
                    setCompleted={setCompleted}
                    setDateTime={setDateTime}
                    setNotifications={setNotifications}
                    setRadioValue={setRadioValue}
                    setSelectValue={setSelectValue}
                    setSliderValue={setSliderValue}
                    sliderValue={sliderValue}
                  />
                ) : null}
                {section.id === "overlays" ? (
                  <OverlaysDemo
                    dialogOpen={dialogOpen}
                    setDialogOpen={setDialogOpen}
                  />
                ) : null}
                {section.id === "navigation" ? (
                  <NavigationDemo
                    selectedNavTab={selectedNavTab}
                    setSelectedNavTab={setSelectedNavTab}
                  />
                ) : null}
                {section.id === "custom" ? (
                  <CustomDemo
                    selectedPackage={selectedPackage}
                    setSelectedPackage={setSelectedPackage}
                  />
                ) : null}
              </GallerySection>
            ))}
    </div>
  );
}

function MetricCard({
  icon: Icon,
  label,
  value,
}: {
  icon: IconComponent;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-subtle bg-card p-4">
      <Row className="items-center justify-between gap-3">
        <div>
          <p className="text-sm text-muted-foreground">{label}</p>
          <p className="mt-1 text-lg font-semibold">{value}</p>
        </div>
        <Centered className="h-10 w-10 rounded-lg bg-muted">
          <Icon className="h-5 w-5 text-muted-foreground" />
        </Centered>
      </Row>
    </div>
  );
}

function GallerySection({
  children,
  components,
  icon: Icon,
  summary,
  title,
}: {
  children: ReactNode;
  components: string[];
  icon: IconComponent;
  summary: string;
  title: string;
}) {
  return (
    <section className="scroll-mt-20">
      <div className="mb-4 flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <Row className="items-center gap-2">
            <Centered className="h-9 w-9 rounded-lg border border-subtle bg-card">
              <Icon className="h-4 w-4" />
            </Centered>
            <h2>{title}</h2>
          </Row>
          <p className="mt-2 max-w-2xl text-muted-foreground">{summary}</p>
        </div>
        <Row className="flex-wrap gap-2">
          {components.map((name) => (
            <Badge key={name} size="sm" variant="secondary">
              {name}
            </Badge>
          ))}
        </Row>
      </div>
      {children}
    </section>
  );
}

function PreviewPanel({
  children,
  description,
  title,
}: {
  children: ReactNode;
  description?: string;
  title: string;
}) {
  return (
    <div className="rounded-xl border border-subtle bg-card">
      <div className="border-b border-subtle p-4">
        <p className="font-semibold">{title}</p>
        {description ? (
          <p className="mt-1 text-sm text-muted-foreground">{description}</p>
        ) : null}
      </div>
      <div className="p-4">{children}</div>
    </div>
  );
}

function FoundationsDemo() {
  return (
    <Grid className="grid-cols-1 gap-4 xl:grid-cols-3">
      <PreviewPanel title="Buttons + Badges">
        <Col className="gap-4">
          <Row className="flex-wrap gap-2">
            <Button>Default</Button>
            <Button variant="secondary">Secondary</Button>
            <Button variant="outline">Outline</Button>
            <Button variant="ghost">Ghost</Button>
            <Button variant="destructive">Destructive</Button>
            <Button size="icon" variant="tertiary" aria-label="Run">
              <Play className="h-4 w-4" />
            </Button>
          </Row>
          <Separator />
          <Row className="flex-wrap gap-2">
            <Badge>Default</Badge>
            <Badge variant="status-running">Running</Badge>
            <Badge variant="status-success">Success</Badge>
            <Badge variant="status-warning">Warning</Badge>
            <Badge variant="status-failure">Failure</Badge>
          </Row>
        </Col>
      </PreviewPanel>

      <PreviewPanel title="Alerts">
        <Col className="gap-3">
          <Alert
            icon={<Info className="h-5 w-5" />}
            title="Token set loaded"
            description="Theme variables are available to Tailwind utilities."
          />
          <AlertInfo
            title="Info variant"
            content="A compact custom alert that keeps icon and copy aligned."
          />
          <AlertWarning
            title="Warning variant"
            content="Useful for migration notes and destructive flows."
          />
          <Alert
            icon={<ShieldAlert className="h-5 w-5" />}
            title="Destructive"
            description="High-friction copy for dangerous actions."
            variant="destructive"
          />
        </Col>
      </PreviewPanel>

      <PreviewPanel title="Cards + Loaders">
        <Col className="gap-4">
          <Card>
            <CardHeader>
              <Row className="items-center justify-between gap-3">
                <div>
                  <CardTitle>Card surface</CardTitle>
                  <CardDescription>
                    Header, content, and footer slots.
                  </CardDescription>
                </div>
                <Avatar>
                  <AvatarFallback>HC</AvatarFallback>
                </Avatar>
              </Row>
            </CardHeader>
            <CardContent>
              <SkeletonCard />
            </CardContent>
            <CardFooter className="gap-2">
              <Spinner />
              <ScaleLoader height={14} width={2} />
            </CardFooter>
          </Card>
          <Row className="items-center gap-3">
            <SkeletonAvatar />
            <Col className="w-full gap-2">
              <SkeletonText className="w-2/3" />
              <SkeletonButton className="w-32" />
            </Col>
          </Row>
        </Col>
      </PreviewPanel>
    </Grid>
  );
}

function InputsDemo({
  calendarDate,
  completed,
  dateTime,
  notifications,
  radioValue,
  selectValue,
  setCalendarDate,
  setCompleted,
  setDateTime,
  setNotifications,
  setRadioValue,
  setSelectValue,
  setSliderValue,
  sliderValue,
}: {
  calendarDate?: Date;
  completed: boolean;
  dateTime?: Date;
  notifications: boolean;
  radioValue: string;
  selectValue: string;
  setCalendarDate: (date: Date | undefined) => void;
  setCompleted: (value: boolean) => void;
  setDateTime: (date: Date | undefined) => void;
  setNotifications: (value: boolean) => void;
  setRadioValue: (value: string) => void;
  setSelectValue: (value: string) => void;
  setSliderValue: (value: number) => void;
  sliderValue: number;
}) {
  return (
    <Grid className="grid-cols-1 gap-4 xl:grid-cols-[1fr_1fr_360px]">
      <PreviewPanel title="Text inputs">
        <Col className="gap-4">
          <Input
            icon={<Search className="h-4 w-4 text-muted-foreground" />}
            label="Search"
            placeholder="Find a model"
            hint="Input supports labels, hints, icons, and error states."
          />
          <Input
            error="Use a shorter display name."
            label="Errored input"
            defaultValue="Production evaluation suite"
          />
          <Textarea
            label="Notes"
            placeholder="Capture an implementation note"
            defaultValue="Copied tokens, primitives, and custom renderers."
          />
          <Label htmlFor="gallery-label-demo">Standalone label</Label>
          <input id="gallery-label-demo" className="sr-only" />
        </Col>
      </PreviewPanel>

      <PreviewPanel title="Choice controls">
        <Col className="gap-5">
          <Checkbox
            checked={completed}
            id="component-check"
            label="Include completed components"
            onCheckedChange={(value) => setCompleted(value === true)}
          />
          <Switch
            checked={notifications}
            id="component-switch"
            label="Send gallery notifications"
            onCheckedChange={setNotifications}
          />
          <RadioGroup value={radioValue} onValueChange={setRadioValue}>
            <Row className="items-center gap-2">
              <RadioGroupItem id="fast" value="fast" />
              <Label htmlFor="fast">Fast</Label>
            </Row>
            <Row className="items-center gap-2">
              <RadioGroupItem id="balanced" value="balanced" />
              <Label htmlFor="balanced">Balanced</Label>
            </Row>
            <Row className="items-center gap-2">
              <RadioGroupItem id="precise" value="precise" />
              <Label htmlFor="precise">Precise</Label>
            </Row>
          </RadioGroup>
          <Col className="gap-2">
            <Row className="items-center justify-between">
              <Label>Confidence</Label>
              <Badge variant="secondary" size="sm">
                {sliderValue}%
              </Badge>
            </Row>
            <Slider
              aria-label="Confidence"
              max={100}
              min={0}
              onValueChange={(value) => setSliderValue(value[0] ?? 0)}
              step={1}
              value={[sliderValue]}
            />
          </Col>
          <Select
            label="Environment"
            onValueChange={setSelectValue}
            options={[
              { label: "Local", value: "local" },
              { label: "Staging", value: "staging" },
              { label: "Production", value: "production" },
            ]}
            placeholder="Pick an environment"
            value={selectValue}
          />
        </Col>
      </PreviewPanel>

      <PreviewPanel title="Calendar + time">
        <Col className="gap-4">
          <div className="rounded-xl border border-subtle bg-background">
            <Calendar
              mode="single"
              onSelect={setCalendarDate}
              selected={calendarDate}
            />
          </div>
          <DateTimePicker
            granularity="minute"
            hourCycle={12}
            onChange={setDateTime}
            value={dateTime}
          />
          <SearchInput
            debounceMs={120}
            onChange={() => undefined}
            placeholder="SearchInput component"
          />
        </Col>
      </PreviewPanel>
    </Grid>
  );
}

function OverlaysDemo({
  dialogOpen,
  setDialogOpen,
}: {
  dialogOpen: boolean;
  setDialogOpen: (open: boolean) => void;
}) {
  return (
    <Grid className="grid-cols-1 gap-4 xl:grid-cols-3">
      <PreviewPanel title="Dialogs">
        <Col className="gap-3">
          <Dialog
            dialogDescription="This wrapper composes Radix dialog primitives with local button variants."
            dialogTitle="Review component changes"
            onConfirm={() => {
              toast.success({
                title: "Confirmed",
                description: "Dialog action completed.",
              });
              setDialogOpen(false);
            }}
            onOpenChange={setDialogOpen}
            open={dialogOpen}
            trigger={<Button>Open dialog</Button>}
          >
            <p className="text-sm text-muted-foreground">
              The modal content can hold forms, status copy, or preview
              summaries.
            </p>
          </Dialog>

          <AlertDialogRoot>
            <AlertDialogTrigger asChild>
              <Button variant="outline">Open alert dialog</Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Archive gallery snapshot?</AlertDialogTitle>
                <AlertDialogDescription>
                  This shows the lower-level AlertDialog primitives directly.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  onClick={() =>
                    toast.warning({
                      title: "Archived",
                      description: "Alert dialog action fired.",
                    })
                  }
                  variant="destructive"
                >
                  Archive
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialogRoot>

          <Prompt
            confirmButtonVariant="destructive"
            confirmText="Reset"
            description="Prompt wraps AlertDialog for terse confirmation flows."
            onConfirm={() =>
              toast.error({
                title: "Reset requested",
                description: "Prompt confirm callback fired.",
              })
            }
            title="Reset gallery state?"
            trigger={<Button variant="destructive">Open prompt</Button>}
          />
        </Col>
      </PreviewPanel>

      <PreviewPanel title="Menus + popovers">
        <Col className="gap-3">
          <Sheet>
            <SheetTrigger asChild>
              <Button variant="secondary">
                <PanelRightOpen className="mr-2 h-4 w-4" />
                Open sheet
              </Button>
            </SheetTrigger>
            <SheetContent>
              <SheetHeader>
                <SheetTitle>Side panel</SheetTitle>
                <SheetDescription>
                  Sheet content uses the same token set as the page.
                </SheetDescription>
              </SheetHeader>
              <div className="my-6 rounded-xl border border-subtle p-4">
                <ScoreBadge scorePercent={96.42} label="Coverage" />
              </div>
              <SheetFooter>
                <Button variant="outline">Close</Button>
              </SheetFooter>
            </SheetContent>
          </Sheet>

          <Popover>
            <PopoverTrigger asChild>
              <Button variant="outline">Open popover</Button>
            </PopoverTrigger>
            <PopoverContent align="start">
              <Col className="gap-3">
                <p className="font-semibold">Popover content</p>
                <Input label="Alias" defaultValue="halo" />
              </Col>
            </PopoverContent>
          </Popover>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline">
                <MoreHorizontal className="mr-2 h-4 w-4" />
                Menu
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-56">
              <DropdownMenuLabel>Gallery actions</DropdownMenuLabel>
              <DropdownMenuItem>
                Refresh demos
                <DropdownMenuShortcut>R</DropdownMenuShortcut>
              </DropdownMenuItem>
              <DropdownMenuCheckboxItem checked>
                Show component badges
              </DropdownMenuCheckboxItem>
              <DropdownMenuSeparator />
              <DropdownMenuRadioGroup value="desktop">
                <DropdownMenuRadioItem value="desktop">
                  Desktop
                </DropdownMenuRadioItem>
                <DropdownMenuRadioItem value="mobile">
                  Mobile
                </DropdownMenuRadioItem>
              </DropdownMenuRadioGroup>
            </DropdownMenuContent>
          </DropdownMenu>
        </Col>
      </PreviewPanel>

      <PreviewPanel title="Tooltips + toast">
        <Col className="gap-4">
          <Tooltip content="Tooltip content follows the copied theme.">
            <Button variant="outline">Hover for tooltip</Button>
          </Tooltip>
          <Row className="flex-wrap gap-2">
            <Button
              onClick={() =>
                toast.info({
                  title: "Info toast",
                  description: "Toast provider is mounted at the app root.",
                })
              }
              variant="secondary"
            >
              Info
            </Button>
            <Button
              onClick={() =>
                toast.success({
                  title: "Success toast",
                  description: "The hook is exported from the local UI package.",
                })
              }
            >
              Success
            </Button>
            <Button
              onClick={() =>
                toast.warning({
                  title: "Warning toast",
                  description: "Transient feedback is working.",
                })
              }
              variant="outline"
            >
              Warning
            </Button>
          </Row>
        </Col>
      </PreviewPanel>
    </Grid>
  );
}

function NavigationDemo({
  selectedNavTab,
  setSelectedNavTab,
}: {
  selectedNavTab: string;
  setSelectedNavTab: (value: string) => void;
}) {
  return (
    <Grid className="grid-cols-1 gap-4 xl:grid-cols-2">
      <PreviewPanel title="Disclosure + tabs">
        <Col className="gap-5">
          <Accordion collapsible defaultValue="tokens" type="single">
            <AccordionItem value="tokens">
              <AccordionTrigger>Theme tokens</AccordionTrigger>
              <AccordionContent>
                Tailwind color, typography, and semantic variables are imported
                from the copied UI package.
              </AccordionContent>
            </AccordionItem>
            <AccordionItem value="fonts">
              <AccordionTrigger>Font assets</AccordionTrigger>
              <AccordionContent>
                Local font files are referenced by relative CSS URLs so the
                desktop view can load them after build.
              </AccordionContent>
            </AccordionItem>
          </Accordion>

          <Tabs defaultValue="preview">
            <TabsList>
              <TabsTrigger value="preview">Preview</TabsTrigger>
              <TabsTrigger value="props">Props</TabsTrigger>
              <TabsTrigger value="state">State</TabsTrigger>
            </TabsList>
            <TabsContent value="preview">
              <Card className="p-4">
                <p className="text-muted-foreground">
                  Tabs expose primitive slots while keeping the visual language.
                </p>
              </Card>
            </TabsContent>
            <TabsContent value="props">
              <Code disableCopyToClipboard>defaultValue="preview"</Code>
            </TabsContent>
            <TabsContent value="state">
              <Badge variant="status-running">Mounted</Badge>
            </TabsContent>
          </Tabs>

          <NavTabs value={selectedNavTab} onValueChange={setSelectedNavTab}>
            <NavTab value="preview" icon={Grid2X2}>
              Preview
            </NavTab>
            <NavTab value="metrics" icon={Activity} count={3}>
              Metrics
            </NavTab>
            <NavTab value="settings" icon={Settings2}>
              Settings
            </NavTab>
          </NavTabs>
        </Col>
      </PreviewPanel>

      <PreviewPanel title="Breadcrumb + table">
        <Col className="gap-5">
          <Breadcrumb>
            <BreadcrumbList>
              <BreadcrumbItem>
                <BreadcrumbLink href="#">HALO</BreadcrumbLink>
              </BreadcrumbItem>
              <BreadcrumbSeparator />
              <BreadcrumbItem>
                <BreadcrumbLink href="#">Library</BreadcrumbLink>
              </BreadcrumbItem>
              <BreadcrumbSeparator />
              <BreadcrumbItem>
                <BreadcrumbPage>Gallery</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>

          <Table>
            <TableCaption>Copied component families</TableCaption>
            <TableHeader>
              <TableRow>
                <TableHead>Family</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right">Exports</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody alternatingRows>
              {gallerySections.slice(0, 4).map((section) => (
                <TableRow key={section.id}>
                  <TableCell>{section.label}</TableCell>
                  <TableCell>
                    <Badge size="sm" variant="status-success">
                      Ready
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right">
                    {section.components.length}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
            <TableFooter>
              <TableRow>
                <TableCell colSpan={2}>Visible families</TableCell>
                <TableCell className="text-right">4</TableCell>
              </TableRow>
            </TableFooter>
          </Table>
        </Col>
      </PreviewPanel>

      <PreviewPanel title="Chart" description="Recharts primitives themed by ChartContainer.">
        <ChartContainer config={chartConfig} className="h-72 w-full">
          <AreaChart data={chartData}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="name" tickLine={false} axisLine={false} />
            <ChartTooltip
              content={
                <ChartTooltipContent
                  indicator="line"
                  valueFormatter={(value) => `${value}`}
                />
              }
            />
            <Area
              dataKey="latency"
              fill="var(--color-latency)"
              fillOpacity={0.18}
              stroke="var(--color-latency)"
              type="monotone"
            />
            <Area
              dataKey="accuracy"
              fill="var(--color-accuracy)"
              fillOpacity={0.12}
              stroke="var(--color-accuracy)"
              type="monotone"
            />
            <ChartLegend content={<ChartLegendContent />} />
          </AreaChart>
        </ChartContainer>
      </PreviewPanel>

      <PreviewPanel title="Chart variant">
        <ChartContainer config={chartConfig} className="h-72 w-full">
          <BarChart data={chartData}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="name" tickLine={false} axisLine={false} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <Bar dataKey="accuracy" fill="var(--color-accuracy)" radius={6} />
          </BarChart>
        </ChartContainer>
      </PreviewPanel>
    </Grid>
  );
}

function CustomDemo({
  selectedPackage,
  setSelectedPackage,
}: {
  selectedPackage: string;
  setSelectedPackage: (value: string) => void;
}) {
  return (
    <Grid className="grid-cols-1 gap-4 xl:grid-cols-3">
      <PreviewPanel title="Identity + states">
        <Col className="gap-5">
          <div className="overflow-hidden rounded-xl border border-subtle p-4">
            <InferenceIcon width="100%" height={34} />
          </div>
          <FakeH1 className="font-newsreader">Gallery headline</FakeH1>
          <Row className="flex-wrap items-center gap-3">
            <PatternAvatarFallback name="Ada Lovelace" patternType="circles" />
            <PatternAvatarFallback name="Grace Hopper" patternType="triangles" />
            <PatternAvatarFallback name="Katherine Johnson" patternType="hexagons" />
          </Row>
          <Row className="flex-wrap gap-2">
            <ScoreBadge scorePercent={100} label="Docs" />
            <ScoreBadge scorePercent={78.34} label="Coverage" />
            <ScoreBadge scorePercent={32.12} label="Risk" />
            <ScoreBadge scorePercent={null} />
          </Row>
          <EmptyState
            action={<Button variant="outline">Create item</Button>}
            description="EmptyState is ready for data-light gallery panels."
            icon={ListChecks}
            title="No rows selected"
          />
        </Col>
      </PreviewPanel>

      <PreviewPanel title="Custom cards + layout">
        <Col className="gap-4">
          <FeatureCard
            card={{
              description: "A copied card wrapper for product feature rows.",
              icon: featureIcon,
              title: "FeatureCard",
            }}
          />
          <div className="relative overflow-hidden rounded-xl border border-subtle bg-card p-5">
            <GradientCard />
            <div className="relative">
              <p className="font-semibold">GradientCard</p>
              <p className="mt-2 text-sm text-muted-foreground">
                Decorative overlay constrained to a preview container.
              </p>
            </div>
          </div>
          <Grid className="gap-3">
            {[
              {
                description: "Buttons, forms, overlays, and data.",
                title: "Core package",
                value: "core",
              },
              {
                description: "Content renderers and brand pieces.",
                title: "Custom package",
                value: "custom",
              },
            ].map(({ description, title, value }) => (
              <SelectableCard
                description={description}
                key={value}
                onClick={() => setSelectedPackage(value)}
                recommended={value === "core"}
                selected={selectedPackage === value}
                title={title}
              />
            ))}
          </Grid>
          <ResponsiveRow className="gap-3 rounded-xl border border-subtle p-3">
            <Row className="items-center gap-2">
              <SquareCheckBig className="h-4 w-4 text-detail-success" />
              <span className="text-sm">Row</span>
            </Row>
            <Col className="gap-1">
              <span className="text-sm">Col</span>
              <span className="text-xs text-muted-foreground">ResponsiveRow</span>
            </Col>
          </ResponsiveRow>
        </Col>
      </PreviewPanel>

      <PreviewPanel title="Code + data">
        <Col className="gap-4">
          <CommandBlock cmd="bun run build:web" />
          <Code showCopyIcon textToCopy="~/lib/ui">
            ~/lib/ui
          </Code>
          <CodeBlock
            code={sampleCode}
            language="tsx"
            obfuscatedCode={sampleCode}
          />
          <JsonComponent
            data={{
              app: "halo",
              copied: true,
              components: gallerySections.length,
            }}
          />
        </Col>
      </PreviewPanel>

      <PreviewPanel title="Mermaid + loading">
        <Col className="gap-4">
          <MermaidDiagram
            caption="Local UI migration flow"
            code={sampleMermaid}
          />
          <div className="relative h-28 overflow-hidden rounded-xl border border-subtle">
            <LoadingScreen />
          </div>
          <Card className="p-4">
            <Row className="items-center gap-2">
              <BadgeCheck className="h-4 w-4 text-detail-success" />
              <span className="text-sm font-medium">
                HomePageBackdrop is copied and intentionally not mounted here.
              </span>
            </Row>
          </Card>
        </Col>
      </PreviewPanel>

      <PreviewPanel title="Layout helpers">
        <Grid className="grid-cols-2 gap-3">
          <Centered className="min-h-24 rounded-xl border border-subtle bg-muted">
            <Col className="items-center gap-2">
              <Grid2X2 className="h-5 w-5" />
              <span className="text-sm">Centered</span>
            </Col>
          </Centered>
          <Grid className="min-h-24 place-items-center rounded-xl border border-subtle bg-muted">
            <Col className="items-center gap-2">
              <FileJson className="h-5 w-5" />
              <span className="text-sm">Grid</span>
            </Col>
          </Grid>
          <Row className="col-span-2 items-center justify-between rounded-xl border border-subtle bg-muted p-4">
            <Row className="items-center gap-2">
              <Terminal className="h-4 w-4" />
              <span className="text-sm">Row</span>
            </Row>
            <Row className="items-center gap-2">
              <ToggleRight className="h-4 w-4" />
              <span className="text-sm">Col</span>
            </Row>
          </Row>
        </Grid>
      </PreviewPanel>
    </Grid>
  );
}
