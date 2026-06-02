type Args = {
  pathname?: string;
};

export function isSiteBannerEnabled({ pathname }: Args): boolean {
  if (pathname?.includes("blog/schematron")) {
    return false;
  }

  // Single production environment
  return true;
}
