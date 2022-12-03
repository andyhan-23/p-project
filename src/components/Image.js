import styled from 'styled-components';
import Responsive from '../container/Responsive';

const ImageBlock = styled.div`
  position: fixed;
  width: 50%;
  height: 50%;
  background: red;
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

const Image = () => {
  return (
    <>
      <ImageBlock>
        <Wrapper>
          <input type="file" placeholder="이미지" />
        </Wrapper>
      </ImageBlock>
    </>
  );
};
export default Image;
