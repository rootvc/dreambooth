import Refresh from "./components/refresh";
import Sessions from "./components/sessions";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export default async function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-5">
      <div>
        <Refresh seconds={5} />
        {/* @ts-expect-error Async Server Component */}
        <Sessions />
      </div>
    </main>
  );
}
