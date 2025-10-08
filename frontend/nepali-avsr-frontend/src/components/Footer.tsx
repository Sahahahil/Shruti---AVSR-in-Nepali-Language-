import "@/styles/components/_footer.scss";

export default function Footer() {
  return (
    <footer className="footer">
      <p>© {new Date().getFullYear()} Nepali AVSR Project | All Rights Reserved</p>
    </footer>
  );
}
