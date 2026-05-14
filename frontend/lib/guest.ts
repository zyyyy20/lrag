const GUEST_ID_KEY = "lrag_debug_guest_id";
const GUEST_SALT_KEY = "lrag_debug_guest_salt";

function getBrowserFingerprintSource(): string {
  if (typeof window === "undefined" || typeof navigator === "undefined") {
    return "server";
  }

  const screenInfo =
    typeof screen === "undefined"
      ? "unknown-screen"
      : `${screen.width}x${screen.height}x${screen.colorDepth}`;
  const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone || "";

  return [
    navigator.userAgent,
    navigator.language,
    navigator.languages?.join(",") ?? "",
    navigator.platform,
    screenInfo,
    timezone,
  ].join("|");
}

function randomId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

async function sha256Hex(input: string): Promise<string> {
  if (
    typeof crypto !== "undefined" &&
    crypto.subtle &&
    typeof TextEncoder !== "undefined"
  ) {
    const bytes = new TextEncoder().encode(input);
    const hash = await crypto.subtle.digest("SHA-256", bytes);
    return Array.from(new Uint8Array(hash))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  }

  return btoa(unescape(encodeURIComponent(input)))
    .replace(/[^a-zA-Z0-9]/g, "")
    .slice(0, 64);
}

export async function getDebugGuestHeaders(): Promise<Record<string, string>> {
  if (typeof window === "undefined") {
    return { "X-Debug-Guest-Id": "server-debug-guest" };
  }

  let guestId = localStorage.getItem(GUEST_ID_KEY);
  let salt = localStorage.getItem(GUEST_SALT_KEY);

  if (!salt) {
    salt = randomId();
    localStorage.setItem(GUEST_SALT_KEY, salt);
  }

  const fingerprint = await sha256Hex(getBrowserFingerprintSource());
  if (!guestId) {
    guestId = await sha256Hex(`${fingerprint}|${salt}`);
    localStorage.setItem(GUEST_ID_KEY, guestId);
  }

  return {
    "X-Debug-Guest-Id": guestId,
    "X-Debug-Device-Fingerprint": fingerprint,
  };
}
