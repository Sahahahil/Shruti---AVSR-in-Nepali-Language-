import "@/styles/components/_footer.scss";

export default function Footer() {
  const year = new Date().getFullYear();

  return (
    <footer className="footer">
      <div className="footer__inner">
        <div className="footer__brand">
          <h4 className="footer__title">Shruti: Nepali AVSR</h4>
          <p className="footer__meta">Major Project BCT 2078</p>
          <p className="footer__copyright">Copyright {year}. All rights reserved.</p>
        </div>

        <div className="footer__contributors">
          <p className="footer__submitted">Submitted by</p>
          <ul className="footer__list">
            <li>KCE078BCT015 - Hasana Manandhar</li>
            <li>KCE078BCT024 - Manav Khatiwada</li>
            <li>KCE078BCT029 - Pranil Krishna Palikhel</li>
            <li>KCE078BCT032 - Sahil Duwal</li>
          </ul>
        </div>
      </div>
    </footer>
  );
}
