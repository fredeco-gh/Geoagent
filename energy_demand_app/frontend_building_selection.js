map.on("click", async (e) => {
  if (map.getZoom() < 15) return;

  const { lat, lng: lon } = e.lngLat;

  const res = await fetch(`/api/building-click?lat=${lat}&lon=${lon}`);
  if (!res.ok) return;

  const result = await res.json();
  if (!result.hit) return;

  const b = result.selected;
  new maplibregl.Popup()
    .setLngLat([b.lon, b.lat])
    .setHTML(`
      <b>Building number: ${b.bygningsnummer}</b><br>
      Type: ${b.bygningstype ?? "unknown"}<br>
      Status: ${b.bygningsstatus ?? "unknown"}<br>
      Distance from click: ${b.distance_m.toFixed(1)} m
    `)
    .addTo(map);
});
