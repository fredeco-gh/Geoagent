// The top bar: sidebar toggle, the folder's simulator chip, the meta line
// (model · session), the warming hint, the context-usage figure, and a "Views"
// button to reopen the canvas after it was closed.

import { useController, useSel } from "../context";
import { ChatIcon, KindIcon, MenuIcon, ViewsIcon } from "../icons";

export function Topbar({ onToggleSidebar }: { onToggleSidebar: () => void }) {
  const controller = useController();
  const sim = useSel((s) => s.sim);
  const details = useSel((s) => s.simDetails);
  const meta = useSel((s) => s.meta);
  const warming = useSel((s) => s.warming);
  const usageLabel = useSel((s) => s.usageLabel);
  const usageTitle = useSel((s) => s.usageTitle);
  const viewCount = useSel((s) => s.viewOrder.length);
  const canvasOpen = useSel((s) => s.canvasOpen);
  const activeView = useSel((s) => s.activeView);
  const openView = useSel((s) => s.openView);
  const closeChat = useSel((s) => s.closeChat);
  const closedViews = useSel((s) => s.closedViews);
  const reopenView = useSel((s) => s.reopenView);

  const simName = (sim && details[sim]?.display_name) || sim;
  const showViews = viewCount > 0 && !canvasOpen;

  return (
    <header className="topbar">
      <button
        className="icon-btn sidebar-toggle"
        title="Show or hide chats"
        aria-label="Toggle sidebar"
        onClick={onToggleSidebar}
      >
        <MenuIcon />
      </button>
      {simName ? (
        <span className="sim-chip" title="This folder's simulator">
          {simName}
        </span>
      ) : null}
      <div className="meta">{meta}</div>
      <div className="actions">
        {warming ? <span className="warming">warming up Julia…</span> : null}
        {usageLabel ? (
          <span className="usage" title={usageTitle}>
            {usageLabel}
          </span>
        ) : null}
        {showViews ? (
          <button className="ghost views-btn" onClick={() => activeView && openView(activeView)}>
            <ViewsIcon /> Views <span className="count">{viewCount}</span>
          </button>
        ) : null}
        {Object.values(closedViews).filter((view) => view.kind === "map").map((view) => (
          <button
            key={view.id}
            className="ghost reopen-btn"
            title={`Reopen ${view.title}`}
            onClick={() => reopenView(view.id)}
          >
            <KindIcon kind={view.kind} /> Reopen {view.title}
          </button>
        ))}
        {viewCount > 0 ? (
          <button className="icon-btn" title="Hide chat" aria-label="Hide chat" onClick={closeChat}>
            <ChatIcon />
          </button>
        ) : null}
        <button
          className="ghost keys-btn"
          title="Set or change provider API keys"
          onClick={() => controller.openKeys()}
        >
          Keys
        </button>
      </div>
    </header>
  );
}
