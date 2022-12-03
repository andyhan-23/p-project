import styled from 'styled-components';
import Responsive from '../container/Header';

const HeaderBlock = styled.div`
  position: relative;
  width: 100%;
  background: white;
  box-shadow: 0px 2px 4px rgba(0, 0, 0.1);
  margin-top: 1rem;
  padding: 1rem;
  position: fixed;
  bottom: 0;
  left: 0;
  width: 100%;
}
`;

const Wrapper = styled(Responsive)`
  height: 4rem;
  display: fixed;

  align-items: center;
  justify-content: center;
  .logo {
    font-size: 1.125rem;
    font-weight: 800;
    letter-spacing: 2px;
  }
`;

const Spacer = styled.div`
  height: 4rem;
`;

const Footer = () => {
  return (
    <>
      <HeaderBlock>
        <Wrapper>
          <div className="logo">방제법</div>
        </Wrapper>
      </HeaderBlock>
      <Spacer />
    </>
  );
};
export default Footer;
