import { Card } from "~/lib/ui/components/ui/Card";
import { cn } from "~/lib/ui/utils/utils";

export type FeatureCard = {
  title: string;
  description: string;
  icon: string;
};

type FeatureCardProps = {
  card: FeatureCard;
  className?: string;
};

export function FeatureCard({ card, className }: FeatureCardProps) {
  return (
    <Card
      className={cn(
        `
          flex flex-col p-6 transition-all

          hover:shadow-sm
        `,
        className,
      )}
    >
      <div className="mb-5">
        <div
          className={`
            flex h-12 w-12 items-center justify-center rounded-full
            bg-card-foreground/10
          `}
        >
          <img
            alt={card.title}
            className={`
              invert

              dark:invert-0
            `}
            src={card.icon}
          />
        </div>
      </div>
      <h3 className="mb-3 font-semibold">{card.title}</h3>
      <p className="text-muted-foreground">{card.description}</p>
    </Card>
  );
}

export type FeatureCardsRowProps = {
  cards: FeatureCard[];
  className?: string;
};
export function FeatureCardsRow({ cards, className }: FeatureCardsRowProps) {
  return (
    <div
      className={cn(
        `
          grid grid-cols-1 gap-4

          md:grid-cols-3
        `,
        className,
      )}
    >
      {cards.map((card, index) => (
        <FeatureCard card={card} key={index} />
      ))}
    </div>
  );
}
