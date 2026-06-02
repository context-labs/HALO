import { useEffect, useState } from "react";
import { SearchIcon } from "lucide-react";

import { Row } from "~/lib/ui/components/custom/Row";
import { Input } from "~/lib/ui/components/ui/Input";
import { cn } from "~/lib/ui/utils/utils";

type SearchInputProps = {
  className?: string;
  onChange: (value: string) => void;
  placeholder?: string;
  defaultValue?: string;
  debounceMs?: number;
};

export function SearchInput({
  className,
  debounceMs = 300,
  defaultValue = "",
  onChange,
  placeholder = "Search...",
}: SearchInputProps) {
  const [value, setValue] = useState(defaultValue);

  useEffect(() => {
    const timer = setTimeout(() => {
      onChange(value);
    }, debounceMs);

    return () => clearTimeout(timer);
  }, [value, onChange, debounceMs]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setValue(newValue);
  };

  return (
    <Row className={cn("relative", className)}>
      <SearchIcon
        className={`
          absolute left-3 top-1/2 z-10 h-4 w-4 -translate-y-1/2
          text-muted-foreground
        `}
      />
      <Input
        className="w-full pl-9 pr-4"
        onChange={handleChange}
        placeholder={placeholder}
        value={value}
      />
    </Row>
  );
}
