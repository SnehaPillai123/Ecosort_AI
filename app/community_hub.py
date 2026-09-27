"""
Community Hub — backed by db.py (SQLite), so it's genuinely shared and
persistent across everyone using the same running app instance, not just
within one browser tab.

Requires a "user_name" to be set in st.session_state (the app asks for a
nickname before showing this hub) — this is a typed nickname, NOT a
verified identity. See db.py's module docstring for the full honest
scope of what this is and isn't.
"""

import os
import time

import streamlit as st

import db
import geo

ISSUE_TYPES = [
    "Illegal dumping",
    "Overflowing / uncollected bin",
    "Bad odor / suspected health hazard",
    "Standing water / pest concern",
    "Other",
]

REWARDS_CATALOG = [
    {"name": "Recycled Paper Notebook", "cost": 40, "made_from": "Reclaimed paper waste", "icon": "📓"},
    {"name": "Upcycled Tote Bag", "cost": 80, "made_from": "Reclaimed fabric offcuts", "icon": "👜"},
    {"name": "Recycled Plastic Pen", "cost": 25, "made_from": "Reclaimed PET plastic", "icon": "🖊️"},
    {"name": "Planter Pot (Recycled Clay)", "cost": 100, "made_from": "Reclaimed ceramic waste", "icon": "🪴"},
]


def render_community_hub(user_name: str):
    st.caption(
        "🗄️ **Live shared data:** everyone using this running app sees the "
        "same reports, events, listings, and leaderboard — backed by a real "
        "database, not just your browser tab. *(Nickname-based, not "
        "authenticated.)*"
    )
    st.write("")

    hub_tab1, hub_tab2, hub_tab3, hub_tab4 = st.tabs(
        ["📸 Report an Issue", "🗓️ Cleanup Events", "🛍️ Eco-Swap Marketplace", "🎁 Rewards Store"]
    )

    # -------------------------------------------------------------
    # Report an issue
    # -------------------------------------------------------------
    with hub_tab1:
        st.markdown("#### Report a waste or sanitation issue")
        st.caption(
            "See overflowing bins, illegal dumping, or a health hazard near "
            "you? Report it here — in a real deployment this would route to "
            "local authorities; verified reports earn Green Points."
        )

        with st.container(border=True):
            with st.form("issue_report_form", clear_on_submit=True):
                issue_photo = st.file_uploader("📷 Photo evidence", type=["jpg", "jpeg", "png"])
                issue_type = st.selectbox("Issue type", ISSUE_TYPES)
                issue_location = st.text_input("📍 Location (area/landmark)")
                issue_notes = st.text_area("Additional details (optional)")
                submitted = st.form_submit_button("🚨 Submit report", use_container_width=True)

        if submitted:
            if not issue_photo or not issue_location.strip():
                st.error("Please attach a photo and enter a location before submitting.")
            else:
                fname = f"{int(time.time())}_{issue_photo.name}"
                fpath = os.path.join(db.UPLOADS_DIR, fname)
                with open(fpath, "wb") as f:
                    f.write(issue_photo.getbuffer())

                coords = geo.geocode(issue_location.strip())  # best-effort; None is fine
                lat, lon = coords if coords else (None, None)

                db.add_issue(user_name, issue_type, issue_location.strip(), issue_notes.strip(), fpath, lat=lat, lon=lon)
                db.log_activity(user_name, "issue", points=5, category=issue_type)
                map_note = " 📍 Pinned on the Impact Map." if coords else ""
                st.success(f"Report submitted — +5 🌿 Green Points. Thank you for keeping your area clean.{map_note}")
                st.rerun()

        issues = db.get_issues(limit=10)
        if issues:
            st.write("")
            st.markdown("##### 🕒 Recent reports (all users, shared)")
            unresolved_ids = {i["id"] for i in db.get_unresolved_issues(limit=200)}
            for issue in issues:
                with st.container(border=True):
                    cols = st.columns([1, 3, 1.3])
                    with cols[0]:
                        if issue["photo_path"] and os.path.exists(issue["photo_path"]):
                            st.image(issue["photo_path"], width=80)
                    with cols[1]:
                        st.markdown(f"**{issue['issue_type']}** at *{issue['location']}*")
                        st.caption(f"🙋 {issue['user_name']} · {issue['status']} · {issue['timestamp']}")
                    with cols[2]:
                        if issue["id"] in unresolved_ids:
                            if st.button("🧹 Organize cleanup", key=f"organize_{issue['id']}"):
                                st.session_state["prefill_event_issue_id"] = issue["id"]
                                st.session_state["prefill_event_location"] = issue["location"]
                                st.session_state["prefill_event_photo"] = issue["photo_path"]
                                st.info("Switch to the **Cleanup Events** tab — this report is pre-filled there.")

    # -------------------------------------------------------------
    # Cleanup events
    # -------------------------------------------------------------
    with hub_tab2:
        st.markdown("#### Upcoming community cleanups")
        st.caption("Shared across everyone using this app — RSVPs are real and counted once per person.")

        upcoming = db.get_events(status="upcoming")
        completed = db.get_events(status="completed")

        for ev in upcoming:
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                with c1:
                    linked = " 🔗 *linked to a community report*" if ev.get("issue_id") else ""
                    badge = "🟡 Demo" if ev.get("created_by") == "Demo" else "🟢 Live"
                    st.markdown(f"**🌿 {ev['title']}**{linked}  ·  {badge}")
                    st.caption(f"📅 {ev['date']}  ·  📍 {ev['location']}  ·  👥 {ev['rsvps']} going")

                    if ev.get("lat") and ev.get("lon"):
                        fc = geo.forecast(ev["lat"], ev["lon"], days=7)
                        match = next((d for d in fc if d["date"] == ev["date"]), None) if fc else None
                        if match:
                            emoji = geo.weather_emoji(match["weather_code"])
                            st.caption(
                                f"{emoji} {match['max_temp_c']:.0f}°C · {match['rain_chance']}% rain — "
                                f"{geo.cleanup_suitability(match['rain_chance'])}"
                            )

                    if ev.get("before_photo_path") and os.path.exists(ev["before_photo_path"]):
                        st.image(ev["before_photo_path"], caption="Before", width=140)
                with c2:
                    already = db.has_rsvped(ev["id"], user_name)
                    if already:
                        st.button("✅ Going", key=f"rsvp_{ev['id']}", disabled=True, use_container_width=True)
                    else:
                        if st.button("🙋 RSVP", key=f"rsvp_{ev['id']}", use_container_width=True):
                            is_new = db.rsvp_event(ev["id"], user_name)
                            if is_new:
                                db.log_activity(user_name, "rsvp", points=3, category=ev["title"])
                                st.success(f"RSVP'd to {ev['title']} — +3 🌿 Green Points")
                            st.rerun()

                with st.expander(f"✅ Mark '{ev['title']}' as completed (upload after-photo)"):
                    st.caption(
                        "Anyone can record completion for this demo — a real deployment would "
                        "restrict this to the organizer or a verified check-in."
                    )
                    after_photo = st.file_uploader(
                        "After photo (proof of cleanup)", type=["jpg", "jpeg", "png"], key=f"after_{ev['id']}"
                    )
                    if st.button("🎉 Mark completed", key=f"complete_{ev['id']}"):
                        if not after_photo:
                            st.error("Please upload an after-photo as proof.")
                        else:
                            fname = f"{int(time.time())}_after_{after_photo.name}"
                            fpath = os.path.join(db.UPLOADS_DIR, fname)
                            with open(fpath, "wb") as f:
                                f.write(after_photo.getbuffer())
                            participants = db.complete_event(ev["id"], fpath, bonus_points=15)
                            st.success(
                                f"🎉 '{ev['title']}' marked complete! "
                                f"+15 🌿 Green Points awarded to {len(participants)} participant(s): "
                                f"{', '.join(participants) if participants else '—'}"
                            )
                            st.rerun()

        st.write("")
        st.markdown("##### ➕ Organize a cleanup")
        prefill_location = st.session_state.pop("prefill_event_location", "")
        prefill_issue_id = st.session_state.pop("prefill_event_issue_id", None)
        prefill_photo = st.session_state.pop("prefill_event_photo", None)
        if prefill_issue_id:
            st.info("Pre-filled from your reported issue — adjust the date/title and create the event.")
            if prefill_photo and os.path.exists(prefill_photo):
                st.image(prefill_photo, caption="Linked report photo", width=140)

        with st.container(border=True):
            with st.form("new_event_form", clear_on_submit=True):
                title = st.text_input("Event title", value=(f"Cleanup at {prefill_location}" if prefill_location else ""))
                date = st.date_input("Date")
                location = st.text_input("Location", value=prefill_location)
                create = st.form_submit_button("🌱 Create event", use_container_width=True)
        if create and title.strip() and location.strip():
            coords = geo.geocode(location.strip())
            lat, lon = coords if coords else (None, None)

            event_id = db.add_event(
                title.strip(), str(date), location.strip(), user_name,
                issue_id=prefill_issue_id, before_photo_path=prefill_photo,
                lat=lat, lon=lon,
            )
            db.rsvp_event(event_id, user_name)
            db.log_activity(user_name, "event_created", points=5, category=title.strip())

            map_note = " 📍 Pinned on the Impact Map." if coords else ""
            st.success(f"'{title}' created and visible to everyone — +5 🌿 Green Points{map_note}")

            if coords:
                fc = geo.forecast(lat, lon, days=7)
                match = next((d for d in fc if d["date"] == str(date)), None) if fc else None
                if match:
                    emoji = geo.weather_emoji(match["weather_code"])
                    st.info(
                        f"{emoji} Forecast for {date}: {match['max_temp_c']:.0f}°C, "
                        f"{match['rain_chance']}% rain chance — {geo.cleanup_suitability(match['rain_chance'])}"
                    )
            st.rerun()

        if completed:
            st.write("")
            st.markdown("##### ✅ Recently completed")
            for ev in completed[:5]:
                with st.container(border=True):
                    st.markdown(f"**{ev['title']}** — completed {ev['completed_at']}")
                    bc, ac = st.columns(2)
                    if ev.get("before_photo_path") and os.path.exists(ev["before_photo_path"]):
                        with bc:
                            st.image(ev["before_photo_path"], caption="Before")
                    if ev.get("after_photo_path") and os.path.exists(ev["after_photo_path"]):
                        with ac:
                            st.image(ev["after_photo_path"], caption="After")

    # -------------------------------------------------------------
    # Marketplace
    # -------------------------------------------------------------
    with hub_tab3:
        st.markdown("#### Low-cost, sustainable swaps")
        st.caption(
            "Community-submitted alternatives to common single-use items — "
            "shared across everyone using this app."
        )

        for item in db.get_marketplace():
            with st.container(border=True):
                st.markdown(f"**🛍️ {item['item']}** — {item['price']}")
                st.caption(f"Replaces: *{item['swaps_for']}*  ·  Listed by: {item['submitted_by']}")

        st.write("")
        st.markdown("##### ➕ List your own swap idea")
        with st.container(border=True):
            with st.form("marketplace_form", clear_on_submit=True):
                item_name = st.text_input("Item name")
                swaps_for = st.text_input("Replaces (what single-use item)")
                price = st.text_input("Approx. price")
                add_item = st.form_submit_button("🌿 Add listing", use_container_width=True)
        if add_item and item_name.strip():
            db.add_marketplace_item(item_name.strip(), swaps_for.strip() or "—", price.strip() or "—", user_name)
            db.log_activity(user_name, "listing", points=5, category=item_name.strip())
            st.success("Listing added for everyone to see — +5 🌿 Green Points")
            st.rerun()

    # -------------------------------------------------------------
    # Rewards store
    # -------------------------------------------------------------
    with hub_tab4:
        st.markdown("#### 🎁 Redeem your Green Points")
        available = db.get_user_points(user_name)
        st.metric("Your Available Green Points", f"🌿 {available}")
        st.caption("Earn points by classifying waste, reporting issues, joining cleanups, or listing swaps.")
        st.write("")

        cols = st.columns(2)
        for i, reward in enumerate(REWARDS_CATALOG):
            with cols[i % 2]:
                with st.container(border=True):
                    st.markdown(f"### {reward['icon']}")
                    st.markdown(f"**{reward['name']}**")
                    st.caption(f"Made from: {reward['made_from']}")
                    st.markdown(f"Cost: **{reward['cost']} 🌿**")
                    affordable = available >= reward["cost"]
                    if st.button(
                        f"Redeem ({reward['cost']} pts)", key=f"redeem_{reward['name']}",
                        disabled=not affordable, use_container_width=True,
                    ):
                        db.redeem_reward(user_name, reward["name"], reward["cost"])
                        st.success(f"Redeemed: {reward['name']}! (In a real deployment, this triggers fulfillment.)")
                        st.rerun()
                    if not affordable:
                        st.caption("Not enough points yet")

        redemptions = db.get_user_redemptions(user_name)
        if redemptions:
            st.write("")
            st.markdown("##### 🧾 Your redemptions")
            for r in redemptions:
                st.caption(f"🎁 {r['reward_name']} — {r['cost']} pts ({r['timestamp']})")
