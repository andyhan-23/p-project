import styled from 'styled-components';
import Responsive from '../container/Responsive';

const DetailBlock = styled.div`
  position: fixed;
  width: 50%;
  height: 50%;
  background: green;
  margin-right: 0 auto;
  right: 0;
`;

const Wrapper = styled(Responsive)`
  height: 4rem;
  display: flex;
  align-items: center;
  justify-content: center;
  .logo {
    font-size: 1.125rem;
    font-weight: 800;
    letter-spacing: 2px;
  }
`;

const Detail = () => {
  return (
    <>
      <DetailBlock>
        <Wrapper>
          <h1>설명</h1>
        </Wrapper>
      </DetailBlock>
    </>
  );
};
export default Detail;
